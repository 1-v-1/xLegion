//! The KV slot cache's state model: the residency arrays and the per-step resolve
//! ([`SlotState::resolve_step`]). The design and why: docs/kv-streaming-porting-notes.md
//! §4.10, and the measured basis: §5.3 (42.5% of decode's selected cells read the KV host
//! tier; the 1K hottest blocks carry 93% of the fetches).//!
//! Position B: on a pool with a host tier, the pool's first `w` block ids are the slot
//! window (VRAM, reserved out of allocation with [`crate::kv::BlockPool::reserve`]); every
//! block a sequence holds is a host-backed identity id. A slot is a copy of one block whose
//! bytes equal the host's (§4.10.3: a block this step writes is never a claim, and
//! append-only sequences never write a block twice, so a copy never goes stale).
//!
//! This module is the model that both other bodies mirror:
//! - the op's CPU implementation runs it as its reference (tests, and the parity oracle);
//! - the CUDA kernels (src/cuda/kv_slots.cu) run the same walk device-side, inside the
//!   captured graph, on the arrays this struct lays out field-for-field.
//!
//! The per-step contract (`resolve_step`), matching §4.10:
//! 1. drain the released-block queue (radix frees; §4.10.6),
//! 2. protect the blocks this step's tokens will write (the spans `pos_s .. pos_s + n_tok`
//!    through the live table; those are never claims and their entries are never rewritten),
//! 3. walk every selected cell: map its table entry to a block (an entry under `w` is a
//!    slot already pointed by an earlier layer this step, §4.10.5), mark a hit's clock bit,
//!    claim each miss's block once,
//! 4. pick victims by second-chance clock: never a slot the live table or this call needs,
//!    never a slot this call just claimed (§4.10's `resolve`),
//! 5. rewrite every walked entry to its block's slot,
//! 6. return the host -> slot copies, in claim order, for the copy kernel.
//!
//! Slot content outlives its residency: an evicted slot keeps its bytes (and `slot_block`)
//! until the slot is claimed again, so a table entry never points at bytes that differ from
//! the host's for the block it names — until the host refills the identity table for the
//! next step, which erases every rewrite (§4.10.5).

/// One pool's cache state. The arrays are the device-side arrays, element for element:
/// `residency` is indexed by block id (the whole pool; host-backed ids only ever appear in
/// tables), `-1` for "not resident", else the slot. `slot_block` is indexed by slot and
/// names the block whose bytes the slot holds — kept past eviction until the slot is
/// claimed again (the doc above). `slot_ref` is the clock's second-chance bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotState {
    /// Slot window width: the pool's block ids `0..w`.
    pub w: usize,
    /// Total pool blocks (the id domain).
    pub n_blocks: usize,
    /// block -> slot, -1: not resident.
    pub residency: Vec<i32>,
    /// slot -> its block's id + 1, 0: never claimed. Kept past eviction (doc above), so it
    /// doubles as "whose bytes are these". (`+ 1` so 0 is "none" without a second array.)
    pub slot_block: Vec<u32>,
    /// slot -> the clock's reference bit.
    pub slot_ref: Vec<u8>,
    /// The clock hand.
    pub hand: usize,
    /// Counters carried on the device `ctl` beside the arrays (hit/miss/evict/protect/
    /// overflow, cumulative; read back by the reporter, the Phase 0 pattern).
    pub ctl: SlotCtl,
}

/// The cache's counters (the slot cache's share of the measurement the Phase 0 hook
/// established; the same numbers the runbook decision reads).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotCtl {
    /// Selected cells served from a resident slot.
    pub hits: u64,
    /// Selected cells whose block was claimed into a slot.
    pub misses: u64,
    /// Selected cells skipped as write-protected (never paged, §4.10.3).
    pub protected_skips: u64,
    /// Slots taken from other blocks.
    pub evictions: u64,
    /// Misses no victim could serve (every slot the step needed): the read stays an
    /// identity read, correct, uncached.
    pub overflow: u64,
}

impl SlotState {
    /// An empty cache over a pool of `n_blocks` with a `w`-block window.
    pub fn new(w: usize, n_blocks: usize) -> Self {
        Self {
            w,
            n_blocks,
            residency: vec![-1; n_blocks],
            slot_block: vec![0; w],
            slot_ref: vec![0; w],
            hand: 0,
            ctl: SlotCtl::default(),
        }
    }

    /// §4.10.6: the blocks freed since last step lose their slots. The slot's bytes stay
    /// (harmless), only the mapping dies — a re-allocated id re-claims (and re-copies)
    /// before anything reads it.
    pub fn drain_released(&mut self, released: &[u32]) {
        for &b in released {
            if let Some(r) = self.residency.get_mut(b as usize) {
                if *r >= 0 {
                    *r = -1;
                }
            }
        }
    }

    /// The block a table entry names: an entry at or above `w` is its own block; one below
    /// `w` is a slot an earlier layer pointed this step, and `slot_block` says whose bytes
    /// it holds (§4.10.5). `0` ("never claimed") cannot be addressed by a well-formed walk
    /// — entries only go below `w` through this model's own writes.
    fn entry_block(&self, entry: i32) -> Option<usize> {
        if entry >= self.w as i32 {
            return Some(entry as usize);
        }
        self.slot_block.get(entry as usize).filter(|&&b| b != 0).map(|&b| b as usize - 1)
    }

    /// One step's resolve; the module docs spell the contract. `table` is the live block
    /// table (filled with identity ids by the host before the step, rewritten here, refilled
    /// next step), `sel` the attention node's own selection rows, `pos` the per-sequence
    /// first position, `n_tok` the tokens each sequence runs this step. Returns the
    /// host -> slot copies, in claim order (the copy kernel's work list; empty after a
    /// fully warm step).
    pub fn resolve_step(
        &mut self,
        table: &mut [i32],
        tab_nb1: usize,
        sel: &[i32],
        sel_nb1: usize,
        rows: usize,
        n_tok: usize,
        pos: &[i32],
    ) -> Vec<(u32, u32)> {
        let n_seqs = (rows / n_tok.max(1)).max(1);
        let cap = sel_nb1.saturating_sub(1);
        debug_assert_eq!(sel_nb1, cap + 1);

        // 2. The write-protected set: every block of every sequence's write span, read off
        // the live table before it is touched (§4.10.3). Entries in a span are identity ids
        // by this same rule (write spans are never claims), so no indirection is needed.
        let mut protected = vec![false; self.n_blocks];
        for s in 0..n_seqs {
            let p0 = *pos.get(s).unwrap_or(&0).max(&0) as usize;
            for p in p0..p0 + n_tok {
                if let Some(&e) = table.get(s * tab_nb1 + p / 64) {
                    if (e as usize) < self.n_blocks {
                        protected[e as usize] = true;
                    }
                }
            }
        }

        // 3. Walk the selections: hits mark their clock bit; each miss's block is claimed
        // once. The positions ascend (the `qsa_attn` op's sel format), so a block's cells
        // concentrate on one entry and consecutive cells dedupe trivially.
        let mut claim: Vec<usize> = Vec::new();
        let mut claiming = vec![false; self.n_blocks];
        // Table cells to re-point after victims are picked (each block's entry, deduped by
        // the ascending walk).
        let mut rewrites: Vec<usize> = Vec::new();
        for row in 0..rows {
            let r = &sel[row * sel_nb1..];
            let cnt = r[0].max(0) as usize;
            let seq = (row / n_tok).min(n_seqs - 1);
            let mut last_blk = usize::MAX;
            for slot in 0..cnt.min(cap) {
                let p = r[1 + slot];
                if p < 0 {
                    continue;
                }
                let blk = seq * tab_nb1 + (p as usize) / 64;
                if blk == last_blk {
                    continue;
                }
                last_blk = blk;
                let Some(b) = self.entry_block(table[blk]) else { continue };
                if protected[b] {
                    self.ctl.protected_skips += 1;
                    continue;
                }
                rewrites.push(blk);
                let r = self.residency[b];
                if r >= 0 {
                    self.slot_ref[r as usize] = 1;
                    self.ctl.hits += 1;
                    continue;
                }
                if !claiming[b] {
                    claiming[b] = true;
                    claim.push(b);
                }
            }
        }

        // 4. Victims: second chance, never a slot the live table references (any entry
        // pointing there reads its bytes), never one this call already handed out. A window
        // with nothing freeable leaves the miss uncached (correct, counted, no stall).
        let mut referenced = vec![false; self.w];
        for t in table.iter().take(n_seqs * tab_nb1) {
            if *t >= 0 && (*t as usize) < self.w {
                referenced[*t as usize] = true;
            }
        }
        let mut copies = Vec::new();
        for &b in &claim {
            let mut taken = None;
            for _ in 0..2 * self.w {
                let v = self.hand;
                self.hand = (self.hand + 1) % self.w.max(1);
                if self.slot_ref[v] == 1 {
                    self.slot_ref[v] = 0;
                    continue;
                }
                if referenced[v] {
                    continue;
                }
                let old = self.slot_block[v];
                if old != 0 {
                    self.ctl.evictions += 1;
                    // The evicted block's mapping dies; the bytes stay until re-claimed.
                    if let Some(r) = self.residency.get_mut(old as usize - 1) {
                        if *r == v as i32 {
                            *r = -1;
                        }
                    }
                }
                taken = Some(v);
                break;
            }
            match taken {
                Some(v) => {
                    self.residency[b] = v as i32;
                    self.slot_block[v] = b as u32 + 1;
                    self.slot_ref[v] = 1;
                    referenced[v] = true;
                    self.ctl.misses += 1;
                    copies.push((b as u32, v as u32));
                }
                None => self.ctl.overflow += 1,
            }
        }

        // 5. Point every walked entry at its block's slot (an identity read where the claim
        // overflowed, the entry left as it stands).
        for blk in rewrites {
            let Some(b) = self.entry_block(table[blk]) else { continue };
            let r = self.residency[b];
            if r >= 0 {
                table[blk] = r;
            }
        }
        copies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NB: usize = 16; // pool blocks
    const W: usize = 2; // slot window

    /// One sequence, one selection row over `width` table cells; the row selects `cells`
    /// (positions), the table holds identity ids `0..` per block.
    struct Sim {
        st: SlotState,
        table: Vec<i32>,
        // Each `step` builds one selection row over the identity-refilled table.
    }

    impl Sim {
        fn new(w: usize) -> Self {
            // One sequence. Identity ids start at `w`: the pool's first `w` ids are the
            // reserved window and never appear in a table (§ BlockPool::reserve) — that is
            // what makes an entry below `w` unambiguously a slot.
            let table = (w as i32..(w + 4) as i32).collect();
            Self { st: SlotState::new(w, NB), table }
        }

        fn step(&mut self, positions: &[i32]) -> Vec<(u32, u32)> {
            // Default write position: past the simulated table, so no block is protected.
            self.step_at(1024, positions)
        }

        fn step_at(&mut self, write_pos: i32, positions: &[i32]) -> Vec<(u32, u32)> {
            let mut sel = vec![0i32; 1 + positions.len().max(1)];
            sel[0] = positions.len() as i32;
            sel[1..1 + positions.len()].copy_from_slice(positions);
            let sel_nb1 = sel.len();
            // Host refill: the table is identity again before every step.
            for (i, t) in self.table.iter_mut().enumerate() {
                *t = (self.st.w + i) as i32;
            }
            let table_nb1 = self.table.len();
            let pos = [write_pos];
            self.st
                .resolve_step(&mut self.table, table_nb1, &sel, sel_nb1, 1, 1, &pos)
        }

        /// The mapping the attention would resolve per cell after the step: the block whose
        /// bytes a read sees, through the rewritten table (a slot entry reads the block
        /// `slot_block` names — whose bytes equal that block's host bytes by §4.10.3).
        fn reads(&self, positions: &[i32]) -> Vec<usize> {
            positions
                .iter()
                .map(|&p| {
                    let e = self.table[p as usize / 64];
                    if (e as usize) < self.st.w {
                        self.st.slot_block[e as usize].saturating_sub(1) as usize
                    } else {
                        e as usize
                    }
                })
                .collect()
        }
    }

    #[test]
    fn claims_then_serves_from_slots() {
        // Three slots: the three blocks of the walk all fit, so step three is fully warm.
        // Identity ids start at w=3: positions 0/64/128 name blocks 3/4/5.
        let mut s = Sim::new(3);
        let copies = s.step(&[0, 128]);
        assert_eq!(copies.len(), 2, "two cold blocks, two copies");
        assert_eq!(s.reads(&[0, 128]), vec![3, 5], "bytes name the right blocks");
        assert_eq!(s.st.ctl.misses, 2);
        let copies = s.step(&[0, 64, 128]); // block 4 cold (a free slot), 3 and 5 hit
        assert_eq!(copies.len(), 1, "only the new block copies");
        assert_eq!(copies[0].0, 4);
        assert_eq!(s.st.ctl.misses, 3);
        assert_eq!(s.st.ctl.evictions, 0, "the third slot was free");
        assert_eq!(s.st.ctl.hits, 2, "the two hits of the second step");
        let res1 = s.st.residency.clone();
        let copies = s.step(&[0, 64, 128]); // fully warm
        assert!(copies.is_empty(), "nothing cold left");
        assert_eq!(s.st.ctl.hits, 5);
        assert_eq!(&s.st.residency[..], &res1[..], "hits never move a mapping");
    }

    #[test]
    fn a_block_this_step_writes_is_never_claimed() {
        let mut s = Sim::new(W);
        // The step's token writes block 2 (pos 0); its selection picks that block's cells
        // and block 4's. §4.10.3: the written block stays an identity read.
        let sel = vec![3, 0, 1, 128];
        for (i, t) in s.table.iter_mut().enumerate() {
            *t = (W + i) as i32;
        }
        let copies = s.st.resolve_step(&mut s.table, 4, &sel, 4, 1, 1, &[0]);
        assert!(copies.iter().all(|&(b, _)| b != 2), "block 2 was written, never copied");
        assert!(copies.iter().any(|&(b, _)| b == 4));
        assert_eq!(s.st.residency[2], -1);
        assert_eq!(s.table[0], 2, "its entry stays identity");
        assert!(s.st.ctl.protected_skips > 0);
    }

    #[test]
    fn second_chance_spares_a_referenced_slot_and_overflows_when_none_are_left() {
        // One slot: claiming block 3 right after block 2 in the same step must not evict
        // block 2 (its entry is being read this very step): the miss overflows instead.
        let mut s = Sim::new(1);
        let copies = s.step(&[64, 128]); // blocks 2 and 3, one slot
        assert_eq!(copies.len(), 1, "the second miss finds no victim");
        assert_eq!(s.st.ctl.overflow, 1);
        let kept = copies[0].0 as usize;
        assert!(s.st.residency[kept] == 0);
        assert_eq!(s.st.residency[if kept == 2 { 3 } else { 2 }], -1);
    }

    #[test]
    fn evicts_an_unreferenced_slot_after_clearing_its_clock_bit() {
        let mut s = Sim::new(1);
        s.step(&[0]); // block 1 into the slot, ref set
        let copies = s.step(&[64]); // block 2: block 1 is no longer referenced
        assert_eq!(copies, vec![(2, 0)]);
        assert_eq!(s.st.ctl.evictions, 1);
        assert_eq!(s.st.residency[1], -1);
        assert_eq!(s.st.residency[2], 0);
    }

    #[test]
    fn a_released_block_loses_its_slot_and_reclaims_on_touch() {
        let mut s = Sim::new(W);
        let copies = s.step(&[0]); // block 2
        assert_eq!(copies[0].0, 2);
        s.st.drain_released(&[2]);
        assert_eq!(s.st.residency[2], -1);
        let copies = s.step(&[0]);
        assert_eq!(copies.len(), 1, "the id re-allocates: re-claim, re-copy");
        assert_eq!(copies[0].0, 2);
        assert!(copies[0].1 < W as u32);
        assert!(s.reads(&[0]) == vec![2]);
    }

    /// Random steps against a host whose blocks carry their own bytes: every read through a
    /// rewritten table names exactly the block the walk asked for (the parity property:
    /// on == off, byte for byte), and the state's invariants hold after every step.
    #[test]
    fn randomized_steps_never_read_wrong_bytes() {
        let mut seed = 0x2545F491u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut s = Sim::new(4);
        for _ in 0..200 {
            let n = 1 + (rnd() % 6) as usize;
            // Positions inside the simulated 4-cell table; block = w + cell.
            let positions: Vec<i32> = (0..n).map(|_| (rnd() % 256) as i32).collect();
            let copies = s.step(&positions);
            let _ = copies;
            assert_eq!(
                s.reads(&positions),
                positions
                    .iter()
                    .map(|&p| s.st.w + (p / 64) as usize)
                    .collect::<Vec<_>>(),
                "a read named the wrong block"
            );
            // Invariants: residency inverts slot_block, counts bounded, slots in range.
            let mut used = 0;
            for b in 0..NB {
                let r = s.st.residency[b];
                if r >= 0 {
                    assert!((r as usize) < 4);
                    assert_eq!(s.st.slot_block[r as usize], b as u32 + 1);
                    used += 1;
                }
            }
            assert!(used <= 4);
            for v in 0..4 {
                if s.st.slot_block[v] != 0 {
                    let b = s.st.slot_block[v] as usize - 1;
                    assert!(b >= W && b < NB, "a slot names a reserved or foreign id");
                }
            }
        }
    }
}

// ---- runtime: the registered pools and the graph hook (the CUDA body: src/cuda/kv_slots.cu) ----

/// One copy segment as the CUDA registry takes it: a pool tensor's base and one block's
/// bytes in it. `KvPool::pool_segments` collects them for a tiered pool.
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Seg {
    base: u64,
    bytes: u64,
}

/// The window the measurement opt-in asks for (`LEGION_DEBUG_KV_SLOTS=<blocks>`); `None`
/// (the default) leaves the mechanism off. Phase 1 develops and A/B-tests behind this
/// switch; §4.7's auto-on + `--no-kv-slot-cache` lands after the A/B on the target
/// machines says so.
pub(crate) fn configured() -> Option<usize> {
    static W: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        let v = std::env::var("LEGION_DEBUG_KV_SLOTS").ok()?;
        let w: usize = v.trim().parse().ok()?;
        (w >= 8).then_some(w)
    })
}

/// The devices with a built slot state (kept host-side so the graph builder can gate).
#[cfg(feature = "cuda")]
fn registered() -> &'static std::sync::Mutex<Vec<usize>> {
    static R: std::sync::OnceLock<std::sync::Mutex<Vec<usize>>> = std::sync::OnceLock::new();
    R.get_or_init(Default::default)
}

/// Build device `device`'s slot state for a pool of `n_blocks` blocks whose first `w` are
/// the reserved window, over `segments` (every tiered attention layer's K and V). Called
/// where the pool's tier is built; a rebuild replaces the state (the ids moved).
#[cfg(feature = "cuda")]
pub(crate) fn register(device: usize, w: usize, n_blocks: usize, cell_space: usize, segments: &[Seg]) {
    // SAFETY: plain C; `segments` outlives the call.
    let code = unsafe {
        legion_kv_slots_ensure(
            device as i32,
            w as i64,
            n_blocks as i64,
            cell_space as i64,
            segments.as_ptr(),
            segments.len() as i32,
            (n_blocks).min(8192).max(w) as i32,
        )
    };
    if code != 0 {
        tracing::warn!(device, "kv slots: cannot build the cache state (error {code})");
        return;
    }
    let mut r = registered().lock().unwrap();
    if !r.contains(&device) {
        r.push(device);
    }
    ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether any pool registered a slot state (set once, read on the release path and by no
/// hot loop).
#[cfg(feature = "cuda")]
static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The engine's release hook: a block the allocator freed loses its slot (§4.10.6). With
/// the mechanism off this is one relaxed load on the allocator's path.
pub(crate) fn release(_id: u32) {
    #[cfg(feature = "cuda")]
    if ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
        // SAFETY: plain C.
        unsafe { legion_kv_slots_release(_id as i32) };
    }
}

/// The per-step resolve node, built between the selection and the attention it feeds
/// (`sparse_paged`): reads the selection rows and positions, rewrites the block table in
/// place and pulls the claimed blocks host -> slot. `None` (no node, an untouched graph)
/// unless the device's state is built. A device-only op: on the CPU there is no host
/// tier, no registration, and the body is a pass-through kept for the op contract.
#[cfg(feature = "cuda")]
pub(crate) fn node<'a>(
    ctx: &'a crate::ggml::Context,
    k: crate::ggml::Tensor<'_>,
    sel: crate::ggml::Tensor<'a>,
    table: crate::ggml::Tensor<'a>,
    pos: crate::ggml::Tensor<'a>,
) -> Option<crate::ggml::Tensor<'a>> {
    let dev = k.cuda_device()?;
    if !registered().lock().unwrap().contains(&dev) {
        return None;
    }
    Some(ctx.external_with(
        &KV_SLOTS,
        crate::ggml::types::I8,
        [1, 1, 1, 1],
        &[sel, table, pos],
        &[dev as i32 + 1],
    ))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn node<'a>(
    _ctx: &'a crate::ggml::Context,
    _k: crate::ggml::Tensor<'_>,
    _sel: crate::ggml::Tensor<'a>,
    _table: crate::ggml::Tensor<'a>,
    _pos: crate::ggml::Tensor<'a>,
) -> Option<crate::ggml::Tensor<'a>> {
    None
}

#[cfg(feature = "cuda")]
static KV_SLOTS: crate::ggml::ExternalOp = crate::ggml::ExternalOp {
    name: "qsa_kv_slots",
    cpu: cpu_slots,
    cuda: Some(cuda_slots),
};

/// The CPU body: a pass-through. There is no host tier and no state on the CPU, and
/// `node` never built this op there; the walk itself is [`SlotState::resolve_step`],
/// tested above and mirrored by the CUDA kernel.
#[cfg(feature = "cuda")]
unsafe fn cpu_slots(_dst: &crate::ggml::ffi::ggml_tensor, _ith: usize, _nth: usize) {}

#[cfg(feature = "cuda")]
unsafe fn cuda_slots(
    dst: &crate::ggml::ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `node`: the attention node's own device operands.
    unsafe {
        let (sel_t, tab_t, pos_t) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let p = crate::ggml::external_params(dst);
        let rows = sel_t.ne[1];
        let n_seqs = tab_t.ne[1].max(1);
        legion_kv_slots_graph(
            sel_t.data as *const i32,
            sel_t.nb[1] / 4,
            tab_t.data as *mut i32,
            tab_t.nb[1] / 4,
            pos_t.data as *const i32,
            rows as i32,
            (rows / n_seqs).max(1) as i32,
            p[0] - 1,
            stream,
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// Builds the device's slot state; see [`register`].
    fn legion_kv_slots_ensure(
        dev: i32,
        w: i64,
        n_blocks: i64,
        cell_space: i64,
        segs: *const Seg,
        n_segs: i32,
        copy_max: i32,
    ) -> i32;
    /// The resolve and the copy, on the attention call's stream.
    fn legion_kv_slots_graph(
        sel: *const i32,
        sel_nb1: i64,
        tab: *mut i32,
        tab_nb1: i64,
        pos: *const i32,
        rows: i32,
        n_tok: i32,
        dev: i32,
        stream: *mut std::os::raw::c_void,
    );
    /// A freed block's slot dies.
    fn legion_kv_slots_release(id: i32);
}
