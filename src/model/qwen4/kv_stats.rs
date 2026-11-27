//! The KV slot cache's Phase 0 measurement ([`register`], the `qsa_kv_stats` [`ExternalOp`];
//! the design and the why: docs/kv-streaming-porting-notes.md).
//!
//! A pool with a host tier reads the host blocks it selects over the link every step. Before
//! building a slot cache on top of that tier, count where the QSA selections actually land:
//! the node reads each step's selection rows and block tables (graph data already), and
//! counts, per host block, the distinct fetches a row would cause, plus per-bucket totals
//! (decode and verify at 0, prompt at 8). The counters are per-device device memory at fixed
//! addresses ([`legion_kv_stats_ensure`] at pool-build time, the registry pattern of
//! [`super::cvec`]), so a captured graph bakes nothing the load could move, and nothing the
//! model reads is written: the bits are the same with the measurement on or off.
//!
//! `LEGION_DEBUG_KV_STATS=1` turns it on: every pool with a host tier registers, the node is
//! built between the selection and the attention ([`node`]), and a reporter thread logs the
//! summary — the per-block concentration is what sizes a cache: a window of S slots absorbs
//! the fetches of the S hottest blocks, so the share of the fetches the top S carry is the
//! link traffic a cache of that size could remove (an upper bound: it ignores replacement).
//! Off (the default) builds no node and registers nothing.

use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::{Mutex, OnceLock};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

use super::qsa_attn::DECODE_TOKENS;

/// The buckets' offsets in the 16 u64 control counters (src/cuda/kv_stats.cu): decode and
/// verify at 0, prompt at 8; within each, [0] cells in VRAM blocks, [1] cells in host
/// blocks, [2] host block fetches (deduplicated per row), [3] rows.
const CTL_BUCKETS: usize = 2;
const CTL_PER_BUCKET: usize = 8;

/// Whether the measurement is on (`LEGION_DEBUG_KV_STATS=1`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LEGION_DEBUG_KV_STATS").is_ok_and(|v| v == "1"))
}

/// One pool's counting parameters: the host range starts at block `hbm` and is `n_host`
/// wide. A pseudo-device (negative) is a host (CPU) pool: unbounded, counted into
/// [`cpu_counts`].
#[derive(Clone, Copy)]
struct Reg {
    hbm: usize,
    n_host: usize,
}

fn regs() -> &'static Mutex<HashMap<i32, Reg>> {
    static R: OnceLock<Mutex<HashMap<i32, Reg>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// The CPU pool's counters (the CPU engine has no host tier: the tests' graphs and the
/// CPU reference of the op land here).
pub(crate) struct CpuCounts {
    pub ctl: [u64; CTL_BUCKETS * CTL_PER_BUCKET],
    pub hist: HashMap<i64, u64>,
}

fn cpu_counts() -> &'static Mutex<Option<CpuCounts>> {
    static C: OnceLock<Mutex<Option<CpuCounts>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// Register the pool a device's steps read: host blocks `hbm_blocks..n_blocks` are what the
/// selections can page over. Called where the pool is built (its tier), does nothing with
/// the measurement off, without a tier, or where the counters cannot be built; a pool
/// resized to a new host size restarts its counters (the ids moved).
pub(crate) fn register(device: usize, n_blocks: usize, hbm_blocks: usize) {
    if !enabled() {
        return;
    }
    let n_host = n_blocks.saturating_sub(hbm_blocks);
    if n_host == 0 {
        return;
    }
    #[cfg(feature = "cuda")]
    // SAFETY: plain C, no borrowed memory.
    let code = unsafe { legion_kv_stats_ensure(device as i32, n_host as i64) };
    #[cfg(not(feature = "cuda"))]
    let code = 0i32;
    if code != 0 {
        tracing::warn!(device, "kv stats: cannot build the counters (error {code})");
        return;
    }
    regs().lock().unwrap().insert(
        device as i32,
        Reg { hbm: hbm_blocks, n_host },
    );
    start_reporter();
}

/// The node between [`super::qsa_attn::select`] and attention: counts this step's selections
/// into the pool's counters. `k` gives the device (like [`super::cvec::project`]). `None`:
/// off, or no pool is registered on that device (no host tier: nothing to measure).
/// The caller expands the returned tensor to run it; it writes nothing the model reads.
pub(crate) fn node<'a>(
    ctx: &'a Context,
    k: Tensor<'_>,
    sel: Tensor<'a>,
    table: Tensor<'a>,
) -> Option<Tensor<'a>> {
    if !enabled() {
        return None;
    }
    let dev = match k.cuda_device() {
        Some(dev) => dev as i32,
        None => -1,
    };
    let reg = *regs().lock().unwrap().get(&dev)?;
    // The prompt kernel walks the same rows; its reads are the prompt's, not decode's.
    let off = if sel.ne()[1] > DECODE_TOKENS { 8 } else { 0 };
    // The params: [hbm, n_host, dev + 1 (0: the CPU body), bucket offset].
    Some(ctx.external_with(
        &KV_STATS,
        types::I8,
        [1, 1, 1, 1],
        &[sel, table],
        &[reg.hbm as i32, reg.n_host.min(i32::MAX as usize) as i32, dev + 1, off as i32],
    ))
}

static KV_STATS: ExternalOp = ExternalOp {
    name: "qsa_kv_stats",
    cpu: cpu_stats,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_stats),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The counting itself, over rows × cap entries: the same walk the CUDA kernel runs
/// (src/cuda/kv_stats.cu), on host slices. Rows are counted at slot 0; a host cell whose
/// row already selected an earlier cell of the same block is not a second fetch (the
/// positions are ascending).
fn count(
    sel: &[i32],
    sel_nb1: usize,
    tab: &[i32],
    tab_nb1: usize,
    rows: usize,
    cap: usize,
    n_tok: usize,
    hbm: i64,
    off: usize,
    from: usize,
    step: usize,
    ctl: &mut [u64],
    hist: &mut HashMap<i64, u64>,
) {
    for i in (from..rows * cap).step_by(step) {
        let (row, slot) = (i / cap, i % cap);
        let r = &sel[row * sel_nb1..];
        if slot == 0 {
            ctl[off + 3] += 1;
        }
        if slot >= r[0].max(0) as usize {
            continue;
        }
        let pos = r[1 + slot];
        if pos < 0 {
            continue;
        }
        let blk = tab[(row / n_tok) * tab_nb1 + pos as usize / 64];
        if blk < 0 {
            continue;
        }
        if (blk as i64) < hbm {
            ctl[off] += 1;
            continue;
        }
        ctl[off + 1] += 1;
        if slot > 0 && r[slot] / 64 == pos / 64 {
            continue;
        }
        *hist.entry(blk as i64).or_default() += 1;
        ctl[off + 2] += 1;
    }
}

unsafe fn cpu_stats(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `node`: host memory of the I32 rows and tables.
    unsafe {
        let (sel_t, tab_t) = (&*dst.src[0], &*dst.src[1]);
        let p = crate::ggml::external_params(dst);
        let (off, rows, cap) = (
            p[3] as usize,
            sel_t.ne[1].max(0) as usize,
            (sel_t.ne[0] - 1).max(0) as usize,
        );
        let (sel_nb1, tab_nb1) = ((sel_t.nb[1] / 4).max(1) as usize, (tab_t.nb[1] / 4) as usize);
        let n_seqs = tab_t.ne[1].max(1) as usize;
        let n_tok = (rows / n_seqs).max(1);
        // The farthest element either walk touches: the last row's count plus its cap, and
        // the last sequence's table row.
        let sel = std::slice::from_raw_parts(
            sel_t.data as *const i32,
            rows.saturating_sub(1) * sel_nb1 + cap + 1,
        );
        let tab = std::slice::from_raw_parts(
            tab_t.data as *const i32,
            (n_seqs - 1) * tab_nb1 + tab_t.ne[0].max(0) as usize,
        );
        let mut ctl = [0u64; CTL_BUCKETS * CTL_PER_BUCKET];
        let mut hist: HashMap<i64, u64> = HashMap::new();
        count(
            sel, sel_nb1, tab, tab_nb1, rows, cap, n_tok, p[0] as i64, off, ith, nth.max(1),
            &mut ctl, &mut hist,
        );
        // The CPU body's rows all share the node's bucket; merge what this thread counted.
        let mut g = cpu_counts().lock().unwrap();
        let c = g.get_or_insert_with(|| CpuCounts {
            ctl: [0; CTL_BUCKETS * CTL_PER_BUCKET],
            hist: HashMap::new(),
        });
        for (a, b) in c.ctl.iter_mut().zip(ctl.iter()) {
            *a += b;
        }
        for (k, v) in hist {
            *c.hist.entry(k).or_default() += v;
        }
    }
}

/// The kernel's parameters (src/cuda/kv_stats.cu).
#[cfg(feature = "cuda")]
#[repr(C)]
struct KvStatsArgs {
    sel: *const i32,
    sel_nb1: i64,
    tab: *const i32,
    tab_nb1: i64,
    cap: i32,
    rows: i32,
    n_tok: i32,
    hbm: i64,
    n_host: i64,
    dev: i32,
    off: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// Builds device `dev`'s counters for `n_host` host blocks (a resize restarts them).
    fn legion_kv_stats_ensure(dev: i32, n_host: i64) -> i32;
    /// Enqueues the counting kernel.
    fn legion_kv_stats_launch(args: *const KvStatsArgs, stream: *mut c_void);
    /// Copies the counters out: 16 ctl values, then the host histogram.
    fn legion_kv_stats_read(dev: i32, ctl: *mut u64, hist: *mut u64) -> i32;
    /// The n_host the counters were built for, -1: none.
    fn legion_kv_stats_n_host(dev: i32) -> i64;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_stats(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `node`: device tensors of the stream's device.
    unsafe {
        let (sel_t, tab_t) = (&*dst.src[0], &*dst.src[1]);
        let p = crate::ggml::external_params(dst);
        let args = KvStatsArgs {
            sel: sel_t.data as *const i32,
            sel_nb1: sel_t.nb[1] / 4,
            tab: tab_t.data as *const i32,
            tab_nb1: tab_t.nb[1] / 4,
            cap: (sel_t.ne[0] - 1) as i32,
            rows: sel_t.ne[1] as i32,
            n_tok: (sel_t.ne[1] / tab_t.ne[1].max(1)).max(1) as i32,
            hbm: p[0] as i64,
            n_host: p[1] as i64,
            dev: p[2] - 1,
            off: p[3],
        };
        legion_kv_stats_launch(&args, stream);
        Ok(())
    }
}

/// A device's counters: the 16 control values and the per-host-block histogram.
pub(crate) struct Snapshot {
    pub ctl: [u64; CTL_BUCKETS * CTL_PER_BUCKET],
    pub hist: Vec<u64>,
}

/// Read one pool's counters (device: the CUDA readback, -1: the CPU body's).
pub(crate) fn snapshot(dev: i32) -> Option<Snapshot> {
    if dev < 0 {
        let g = cpu_counts().lock().unwrap();
        let c = g.as_ref()?;
        let mut hist: Vec<(i64, u64)> = c.hist.iter().map(|(k, v)| (*k, *v)).collect();
        hist.sort();
        return Some(Snapshot {
            ctl: c.ctl,
            hist: hist.into_iter().map(|(_, v)| v).collect(),
        });
    }
    let reg = *regs().lock().unwrap().get(&dev)?;
    #[cfg(feature = "cuda")]
    {
        let mut ctl = [0u64; CTL_BUCKETS * CTL_PER_BUCKET];
        let mut hist = vec![0u64; reg.n_host];
        // SAFETY: plain C buffers.
        let code =
            unsafe { legion_kv_stats_read(dev, ctl.as_mut_ptr(), hist.as_mut_ptr()) };
        if code != 0 {
            return None;
        }
        Some(Snapshot { ctl, hist })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = reg;
        None
    }
}

/// The summary one log line carries: per bucket, the rows, the cells the link does and does
/// not carry today, the fetches a slot pool would move, and the share of them the top S
/// blocks already cover (the cache's benefit upper bound at that size).
pub(crate) fn report(dev: i32, s: &Snapshot) {
    let mut sorted: Vec<u64> = s.hist.iter().copied().filter(|c| *c > 0).collect();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let fetches: u64 = sorted.iter().sum();
    let cover = |slots: usize| -> u64 {
        let top: u64 = sorted.iter().take(slots).sum();
        if fetches == 0 { 0 } else { top * 100 / fetches }
    };
    for (b, name) in [(0usize, "decode"), (8usize, "prompt")] {
        let (local, host, fetch, rows) = (s.ctl[b], s.ctl[b + 1], s.ctl[b + 2], s.ctl[b + 3]);
        if rows == 0 {
            continue;
        }
        tracing::info!(
            dev,
            bucket = name,
            rows,
            cells_vram = local,
            cells_host = host,
            fetches = fetch,
            unique = sorted.len(),
            cover_1k = cover(1024),
            cover_8k = cover(8192),
            "kv stats: {name} steps read {host} of {} selected cells from host blocks \
             ({fetch} block fetches, {} unique; a 1K-slot window would carry {}% of them, \
             8K: {}%)",
            local + host,
            sorted.len(),
            cover(1024),
            cover(8192),
        );
    }
}

/// The reporter: logs every registered pool's counters every 30 s (a measurement run reads
/// the trend; the counters are cumulative since the last build of that size).
fn start_reporter() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("kv-stats".to_owned()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(30));
            let devs: Vec<i32> =
                regs().lock().unwrap().keys().copied().filter(|d| *d >= 0).collect();
            for dev in devs {
                if let Some(s) = snapshot(dev) {
                    report(dev, &s);
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The walk over a fixture of two rows: host cells count per read and once per
    /// (row, block) fetch; VRAM cells, negative positions and unmapped blocks only count
    /// where they belong; rows count even when empty.
    #[test]
    fn counts_host_cells_fetches_and_rows() {
        // table row (width 4) for the one sequence: block id by position/64.
        let tab = vec![8, 3, -1, -1, -1, -1, -1, -1];
        // [count, positions…] × 2 rows; cap 4, hbm 8: block 8 is host, block 3 is VRAM.
        let sel = vec![3, 0, 1, 64, -1, 0, -1, -1, -1, -1];
        let mut ctl = [0u64; 16];
        let mut hist: HashMap<i64, u64> = HashMap::new();
        count(&sel, 4, &tab, 4, 2, 4, 1, 8, 0, 0, 1, &mut ctl, &mut hist);
        assert_eq!(ctl[3], 2, "both rows counted");
        assert_eq!(ctl[1], 2, "two selected cells in host blocks");
        assert_eq!(ctl[0], 1, "one selected cell in a VRAM block");
        assert_eq!(ctl[2], 1, "the row's two host cells are one block fetch");
        assert_eq!(hist.get(&8), Some(&1));
        assert_eq!(hist.len(), 1);
    }

    /// Rows map to sequences as the attention reads them (`row / n_tok`), and each sequence
    /// walks its own table row.
    #[test]
    fn rows_follow_their_sequence_table() {
        // Two sequences (width 2): seq 0 reads block 9 first, seq 1 block 10 first.
        let tab = vec![9, -1, 10, -1];
        // Rows are four elements wide: rows 0 and 1 belong to seq 0, rows 2 and 3 to seq 1.
        let sel = vec![
            1, 0, -1, -1, // row 0: block 9 (host)
            1, 63, -1, -1, // row 1: block 9 again (its own fetch)
            1, 0, -1, -1, // row 2: block 10 (host, seq 1's row)
            1, 64, -1, -1, // row 3: seq 1's second block: unmapped, counted nowhere
        ];
        let mut ctl = [0u64; 16];
        let mut hist: HashMap<i64, u64> = HashMap::new();
        count(&sel, 4, &tab, 2, 4, 4, 2, 8, 8, 0, 1, &mut ctl, &mut hist);
        assert_eq!(ctl[8 + 3], 4);
        assert_eq!(ctl[8 + 1], 3, "one cell lands on the unmapped table entry");
        assert_eq!(ctl[8 + 2], 3, "each row fetches its own sequence's block once");
        assert_eq!(hist.get(&9), Some(&2));
        assert_eq!(hist.get(&10), Some(&1));
        assert_eq!(hist.len(), 2);
    }

    /// A step split over threads (`nth > 1`) sums to the same totals as one thread, the way
    /// ggml calls the CPU body.
    #[test]
    fn thread_split_matches_single() {
        // Width 4; blocks 8 and 9 are host, block 3 is VRAM, entry 3 unmapped.
        let tab = vec![8, 3, 9, -1];
        let sel = vec![
            3, 0, 1, 128, -1, // row 0: 8 (once), 8 dedup, 9
            3, 64, 65, 192, -1, // row 1: VRAM twice, then the unmapped entry
        ];
        let mut ctl1 = [0u64; 16];
        let mut hist1: HashMap<i64, u64> = HashMap::new();
        count(&sel, 5, &tab, 4, 2, 4, 2, 8, 0, 0, 1, &mut ctl1, &mut hist1);
        let mut ctl4 = [0u64; 16];
        let mut hist4: HashMap<i64, u64> = HashMap::new();
        for ith in 0..4 {
            count(&sel, 5, &tab, 4, 2, 4, 2, 8, 0, ith, 4, &mut ctl4, &mut hist4);
        }
        assert_eq!(ctl1, ctl4);
        assert_eq!(hist1, hist4);
        assert_eq!(ctl1[0], 2, "the VRAM cells");
        assert_eq!(ctl1[1], 3, "the host cells");
        assert_eq!(ctl1[2], 2, "fetches of blocks 8 and 9 once each");
        assert_eq!(ctl1[3], 2);
    }
}
