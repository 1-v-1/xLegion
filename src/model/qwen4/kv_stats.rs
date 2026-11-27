//! The KV slot cache's Phase 0 measurement ([`register`], [`after`]; the design and the
//! why: docs/kv-streaming-porting-notes.md).
//!
//! A pool with a host tier reads the host blocks it selects over the link every step. Before
//! building a slot cache on top of that tier, count where the QSA selections actually land.
//! [`after`] runs with [`super::qsa_attn`]'s `after_attn` — next to the selection dump of
//! [`super::qsa_attn::sel_dump`], the same shape: one extra kernel on the step's stream,
//! behind nothing the model reads, reading only the selection rows and the block tables.
//! No graph node, no tensor: the graph is the same graph with the measurement on or off.
//!
//! The counters are per-device device memory at fixed addresses ([`register`] builds them
//! where the pool's tier is, the registry pattern of [`super::cvec`]), so a captured graph
//! bakes no address the load could move; a reporter thread logs the summary — the per-block
//! concentration is what sizes a cache: a window of S slots absorbs the fetches of the S
//! hottest blocks, so the share of the fetches the top S carry is the link traffic a cache
//! of that size could remove (an upper bound: it ignores replacement). Off (the default)
//! enqueues no kernel and registers nothing.
//!
//! What the counters say is spelled out in src/cuda/kv_stats.cu: per bucket (decode and
//! verify, prompt), the selected cells in VRAM blocks and in host blocks, the host block
//! fetches deduplicated per row, the rows, and the per-host-block fetch histogram.

use std::sync::OnceLock;
#[cfg(any(test, feature = "cuda"))]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::os::raw::c_void;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

/// The buckets' offsets in the 16 u64 control counters (src/cuda/kv_stats.cu): decode and
/// verify at 0, prompt at 8; within each, [0] cells in VRAM blocks, [1] cells in host
/// blocks, [2] host block fetches (deduplicated per row), [3] rows.
#[cfg(feature = "cuda")]
const CTL_BUCKETS: usize = 2;
#[cfg(feature = "cuda")]
const CTL_PER_BUCKET: usize = 8;

/// Whether the measurement is on (`LEGION_DEBUG_KV_STATS=1`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LEGION_DEBUG_KV_STATS").is_ok_and(|v| v == "1"))
}

/// The registered devices. The kernel's parameters (boundary, width) live with the counters
/// on the CUDA side, so the graph and the launch need nothing else.
#[cfg(feature = "cuda")]
fn regs() -> &'static Mutex<Vec<i32>> {
    static R: OnceLock<Mutex<Vec<i32>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// Register the pool a device's steps read: host blocks `hbm_blocks..n_blocks` are what the
/// selections can page over. Called where the pool is built (its tier), does nothing with
/// the measurement off or without a tier; a pool resized to a new host size restarts its
/// counters (the ids moved). The host tier, and this measurement, are CUDA-only.
pub(crate) fn register(device: usize, n_blocks: usize, hbm_blocks: usize) {
    #[cfg(feature = "cuda")]
    register_impl(device, n_blocks, hbm_blocks);
    #[cfg(not(feature = "cuda"))]
    let _ = (device, n_blocks, hbm_blocks);
}

#[cfg(feature = "cuda")]
fn register_impl(device: usize, n_blocks: usize, hbm_blocks: usize) {
    if !enabled() {
        return;
    }
    let n_host = n_blocks.saturating_sub(hbm_blocks);
    if n_host == 0 {
        return;
    }
    // SAFETY: plain C, no borrowed memory.
    let code = unsafe { legion_kv_stats_ensure(device as i32, n_host as i64, hbm_blocks as i64) };
    if code != 0 {
        tracing::warn!(device, "kv stats: cannot build the counters (error {code})");
        return;
    }
    let mut r = regs().lock().unwrap();
    if !r.contains(&(device as i32)) {
        r.push(device as i32);
    }
    drop(r);
    n_hosts().lock().unwrap().insert(device as i32, n_host);
    start_reporter();
}

/// The step's counting, from `after_attn`: `sel`/`table` the attention node's own operands
/// (rows of `sel_row` I32s `[count, positions ascending, -1]`, tables of `table_width`),
/// `off` the bucket ([0: decode and verify, 8: prompt]).
///
/// # Safety
/// The pointers are device memory of the stream's device, `rows` rows, as the attention
/// launch itself reads them.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn after(
    sel: *const i32,
    sel_row: usize,
    table: *const i32,
    table_width: usize,
    rows: usize,
    n_tok: usize,
    off: i32,
    stream: *mut c_void,
) {
    // SAFETY: the caller's; the kernel reads them as the attention kernel does.
    unsafe {
        legion_kv_stats_after(sel, sel_row as i64, table, table_width as i64, rows as i32,
                              n_tok as i32, off, stream);
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// Builds device `dev`'s counters for the host range starting at block `hbm`, `n_host`
    /// wide (a resize to a new width restarts them).
    fn legion_kv_stats_ensure(dev: i32, n_host: i64, hbm: i64) -> i32;
    /// Enqueues the counting kernel on `stream`.
    fn legion_kv_stats_after(sel: *const i32, sel_nb1: i64, tab: *const i32, tab_nb1: i64,
                             rows: i32, n_tok: i32, off: i32, stream: *mut c_void);
    /// Copies the counters out: 16 ctl values, then the host histogram.
    fn legion_kv_stats_read(dev: i32, ctl: *mut u64, hist: *mut u64) -> i32;
}

/// A device's counters: the 16 control values and the per-host-block histogram.
#[cfg(feature = "cuda")]
pub(crate) struct Snapshot {
    pub ctl: [u64; CTL_BUCKETS * CTL_PER_BUCKET],
    pub hist: Vec<u64>,
}

/// Read one pool's counters.
#[cfg(feature = "cuda")]
pub(crate) fn snapshot(dev: i32) -> Option<Snapshot> {
    let n = read_n_host(dev)?;
    let mut ctl = [0u64; CTL_BUCKETS * CTL_PER_BUCKET];
    let mut hist = vec![0u64; n];
    // SAFETY: plain C buffers, the width the ensure built.
    let code = unsafe { legion_kv_stats_read(dev, ctl.as_mut_ptr(), hist.as_mut_ptr()) };
    if code != 0 {
        return None;
    }
    Some(Snapshot { ctl, hist })
}

/// The `n_host` each registered device's counters were built for (mirrored here so the
/// readback can size its buffer; the kernel's parameters stay on the CUDA side).
#[cfg(feature = "cuda")]
fn n_hosts() -> &'static Mutex<HashMap<i32, usize>> {
    static N: OnceLock<Mutex<HashMap<i32, usize>>> = OnceLock::new();
    N.get_or_init(Default::default)
}

#[cfg(feature = "cuda")]
fn read_n_host(dev: i32) -> Option<usize> {
    n_hosts().lock().unwrap().get(&dev).copied()
}

/// The summary one log line carries: per bucket, the rows, the cells the link does and does
/// not carry today, the fetches a slot pool would move, and the share of them the top S
/// blocks already cover (the cache's benefit upper bound at that size).
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
fn start_reporter() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("kv-stats".to_owned()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(30));
            let devs: Vec<i32> = regs().lock().unwrap().clone();
            for dev in devs {
                if let Some(s) = snapshot(dev) {
                    report(dev, &s);
                }
            }
        });
    });
}

/// The walk src/cuda/kv_stats.cu's kernel runs, in Rust, as the tests' spec of it: over
/// rows × cap entries, the host cells count per read and once per (row, block) fetch (the
/// positions are ascending, so the preceding cell of the same row says this block was
/// already fetched for this token); VRAM cells, negative positions and unmapped blocks
/// count only where they belong; rows count even when empty.
#[cfg(test)]
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

    /// A step split over threads (`nth > 1`) sums to the same totals as one thread: the
    /// walk is per entry, so a kernel grid or a ggml thread split gives the same answer.
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
