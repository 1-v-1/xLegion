//! The KV slot cache's state model: the residency arrays and the per-step resolve
//! ([`SlotState::resolve_step`]). The design and why: docs/kv-streaming-porting-notes.md
//! §4.10, and the measured basis: §5.3 (42.5% of decode's selected cells read the KV host
//! tier; the 1K hottest blocks carry 93% of the fetches).
//!
//! The window lives **inside** the device range `--kv-cache-tokens` already opened: the pool's
//! first `w` block ids are reserved out of allocation ([`crate::kv::BlockPool::reserve`]), so
//! no sequence is ever handed one, and neither capacity flag changes meaning. What it costs is
//! KV capacity - sequences share `n_blocks - w` blocks instead of `n_blocks` - and nothing
//! else: no extra VRAM, no extra pinned RAM, and the tier boundary stays where the user put
//! it, so recent blocks keep landing in device memory on their own (§4.10.2).
//! A slot is a copy of one block whose bytes equal the authority's: a block this step writes
//! is never a *claim*, and if it is already resident this step's copy refreshes it from the
//! authority (the write-through §4.10.1 dropped; taken here as a whole-block byte copy, so a
//! copy matches what the write kernel stored bit for bit and never goes stale).
//!
//! This module is the model that both other bodies mirror:
//! - the op's CPU implementation runs it as its reference (tests, and the parity oracle);
//! - the CUDA kernels (src/cuda/kv_slots.cu) run the same walk device-side, inside the
//!   captured graph, on the arrays this struct lays out field-for-field.
//!
//! The per-step contract (`resolve_step`):
//! 1. drain the released-block queue (radix frees; §4.10.6) - a recycled id must not answer
//!    with its previous owner's slot,
//! 2. the blocks this step writes are never claimed (§4.10.3), and a resident one has its slot
//!    refreshed from the authority first: that write-through is what keeps a copy from going
//!    stale (§4.10.1),
//! 3. walk the selections: a block whose slot holds it is a hit; only a host-backed miss
//!    claims one - a block already in device memory crosses no link, so paging it would spend
//!    a slot and a full copy on data that was never slow,
//! 4. pick victims by second-chance clock, never a slot whose copy this step queued or which
//!    it just handed out,
//! 5. return the authority -> slot copies, in order, for the copy kernel - which compares what
//!    it wrote, because "a slot's bytes equal the authority's" is the single property the whole
//!    design rests on.
//!
//! The block table is never written. `residency` is the map (block -> slot, -1 for "read the
//! authority") and the attention consults it, exactly as Strata's readers go through
//! `page_table`. Rewriting the shared table in place - the first attempt - forced an epoch to
//! know which slots were in use, a scan of the whole table at every layer, and an indirection
//! back through `slot_block`, all to keep a tensor other code reads coherent; and it was a
//! performance problem on its own at a million-token context (§4.10.12).

/// One pool's cache state. The arrays are the device-side arrays, element for element:
/// `residency` is indexed by block id (the whole pool; host-backed ids only ever appear in
/// tables), `-1` for "not resident", else the slot. `slot_block` is indexed by slot and
/// names the block whose bytes the slot holds — kept past eviction until the slot is
/// claimed again (the doc above). `slot_ref` is the clock's second-chance bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotState {
    /// Slot window width: the pool's block ids `0..w`.
    pub w: usize,
    /// Blocks below this are in device memory already and are never paged: a copy of one
    /// crosses no link, and it would spend a slot that a host block needed (§4.10.12).
    pub hbm: usize,
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
            hbm: 0,
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

    /// One step's resolve; the module docs spell the contract. `table` is read only: its
    /// entries are block ids, and the cache is `residency` (block -> slot, -1 = read the
    /// authority), which the attention consults. Rewriting the table in place instead - what
    /// this did first - meant an epoch to know which slots were in use, a scan of the whole
    /// table at every layer, and an indirection back through `slot_block`, all only to keep a
    /// tensor shared with other readers coherent; §4.10.12. Returns the authority -> slot
    /// copies, in order, for the copy kernel.
    pub fn resolve_step(
        &mut self,
        table: &[i32],
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
        let mut held = vec![false; self.w];

        // 1. The blocks this step writes: never claimed (§4.10.3), and a resident one has its
        // slot refreshed - the write-through that keeps a copy from going stale (G4), taken as
        // a whole-block byte copy so it matches `set_rows` bit for bit. The refresh leads the
        // copy list and is never dropped for a claim; where the list cannot hold one, the
        // mapping dies and that read falls back to the authority.
        let mut protected = vec![false; self.n_blocks];
        let mut copies: Vec<(u32, u32)> = Vec::new();
        for s in 0..n_seqs {
            let p0 = *pos.get(s).unwrap_or(&0).max(&0) as usize;
            for p in p0..p0 + n_tok {
                let Some(&e) = table.get(s * tab_nb1 + p / 64) else { continue };
                if e < 0 || e as usize >= self.n_blocks {
                    continue;
                }
                let b = e as usize;
                if std::mem::replace(&mut protected[b], true) {
                    continue;
                }
                let r = self.residency[b];
                if r >= 0 {
                    held[r as usize] = true;
                    copies.push((b as u32, r as u32));
                }
            }
        }

        // 2. The walk: the positions ascend, so one block's cells concentrate on a table entry
        // and consecutive ones dedupe. A resident block is served from its slot; only a
        // host-backed miss claims one.
        let mut claim: Vec<usize> = Vec::new();
        let mut claiming = vec![false; self.n_blocks];
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
                let Some(&e) = table.get(seq * tab_nb1 + (p as usize) / 64) else { continue };
                if e < 0 || e as usize >= self.n_blocks {
                    continue;
                }
                let b = e as usize;
                if b == last_blk || protected[b] || b < self.hbm {
                    continue;
                }
                last_blk = b;
                let v = self.residency[b];
                if v >= 0 {
                    held[v as usize] = true;
                    self.slot_ref[v as usize] = 1;
                    self.ctl.hits += 1;
                    continue;
                }
                if !claiming[b] {
                    claiming[b] = true;
                    claim.push(b);
                }
            }
        }

        // 3. Victims: second chance, never a slot this step reads (its copy is queued, or its
        // attention is about to ask for it) or already handed out. A window with nothing
        // freeable leaves the miss uncached: correct, counted, and no stall.
        for &b in &claim {
            let mut taken = None;
            for _ in 0..2 * self.w {
                let v = self.hand;
                self.hand = (self.hand + 1) % self.w.max(1);
                if self.slot_ref[v] == 1 {
                    self.slot_ref[v] = 0;
                    continue;
                }
                if held[v] {
                    continue;
                }
                if let Some(old) = self.slot_block[v].checked_sub(1) {
                    self.ctl.evictions += 1;
                    // The evicted block falls back to the authority; the bytes stay until the
                    // slot is claimed again, so a concurrent reader never sees a half-written
                    // block.
                    if self.residency[old as usize] == v as i32 {
                        self.residency[old as usize] = -1;
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
                    held[v] = true;
                    self.ctl.misses += 1;
                    copies.push((b as u32, v as u32));
                }
                None => self.ctl.overflow += 1,
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

        /// What a read of each position answers: the authority block, unless the remap sends
        /// it to a slot - in which case the block that slot actually holds. So this asserts
        /// the one property that matters: a remapped read still names the block asked for.
        fn reads(&self, positions: &[i32]) -> Vec<usize> {
            positions
                .iter()
                .map(|&p| {
                    let b = self.st.w + p as usize / 64;
                    let v = self.st.residency[b];
                    if v < 0 {
                        b
                    } else {
                        self.st.slot_block[v as usize].saturating_sub(1) as usize
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

    /// A block already in device memory crosses no link: paging it would spend a window slot
    /// and a whole-block copy on bytes that were never slow, and evict a host block that did
    /// need one.
    #[test]
    fn device_resident_blocks_are_never_paged() {
        let mut s = Sim::new(W);
        s.st.hbm = W as i32 as usize + 3; // only the block past that is host-backed
        // Positions 0 and 192 name table cells 0 and 3: block w and block w+3, the last of
        // which is the only one past `hbm`.
        let copies = s.step(&[0, 192]);
        assert_eq!(copies, vec![(W as u32 + 3, 0)], "only the host-backed block is paged");
        assert_eq!(s.st.residency[W], -1, "the device-backed block keeps its own rows");
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
        assert_eq!(s.st.residency[2], -1, "a block being written is never cached");
        assert_eq!(s.reads(&[0, 1]), vec![2, 2], "and is read from its own rows");
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

    /// A resident block the step writes is refreshed from the authority and stays readable
    /// from its slot: the write-through that lets the newest KV live in the window at all.
    #[test]
    fn a_resident_block_the_step_writes_is_refreshed_and_read_from_its_slot() {
        let mut s = Sim::new(W);
        // Warm it up: block 2 into a slot, with the step's write elsewhere.
        let copies = s.step(&[0]);
        assert_eq!(copies.len(), 1);
        let slot = copies[0].1;
        // Now the step writes block 2 *and* selects it: no claim, but the slot is refreshed
        // and the entry points at it.
        let copies = s.step_at(0, &[0]);
        assert!(
            copies.contains(&(2, slot)),
            "the resident span block's copy is in this step's work list"
        );
        assert_eq!(s.st.residency[2], slot as i32, "still resident, same slot");
        assert_eq!(s.reads(&[0]), vec![2], "read through the refreshed slot");
        assert_eq!(s.st.ctl.overflow, 0);
    }

    /// A refreshed slot is off-limits to this call's victims: two writes and a cold
    /// selection in one step must not hand the refreshed slot to someone else.
    #[test]
    fn a_refreshed_slot_is_never_a_victim_in_the_same_step() {
        let mut s = Sim::new(1);
        let copies = s.step(&[0]); // block 1 owns the only slot
        assert_eq!(copies.len(), 1);
        // The step writes block 1 (refresh) and misses on block 2: the one slot is taken by
        // the refresh, so block 2 overflows rather than evicting it.
        let copies = s.step_at(0, &[0, 64]);
        assert_eq!(
            copies.iter().filter(|&&(b, _)| b == 1).count(),
            1,
            "block 1 refreshed"
        );
        assert!(copies.iter().all(|&(b, _)| b != 2), "block 2 claims nothing");
        assert_eq!(s.st.residency[1], 0, "the refreshed slot survived");
        assert_eq!(s.st.residency[2], -1);
    }

    /// A selection row in the `qsa_attn` format: the count, then the positions.
    fn row(cells: &[i32]) -> Vec<i32> {
        let mut r = vec![0i32; 1 + cells.len()];
        r[0] = cells.len() as i32;
        r[1..].copy_from_slice(cells);
        r
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
            // A quarter of the steps write inside the table: the refreshed blocks are read
            // back through the same table, so a refresh that went to the wrong slot shows up
            // as a wrong read.
            let wp = if rnd() % 4 == 0 { (rnd() % 256) as i32 } else { 1024 };
            let copies = s.step_at(wp, &positions);
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
/// bytes in it. Built per tiered attention layer's K and V by the executor.
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Seg {
    pub(crate) base: u64,
    pub(crate) bytes: usize,
}

/// What the window reports: the counters the resolve kernel keeps, summed over the
/// registered devices. `protected` is the read-side half of the write-through story -
/// selections of a block this step writes that had no slot to read - and `drops` is the
/// releases a full ring could not queue, each of which forces the whole window down.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SlotCounters {
    pub hits: u64,
    pub misses: u64,
    pub protected: u64,
    pub evictions: u64,
    pub overflow: u64,
    pub drops: u64,
    /// The first invariant the kernel broke: `(kind, a, b)`, kind 0 for none. A non-zero
    /// kind is a wrong-read candidate, not a tuning signal - see the kinds in kv_slots.cu.
    pub bad: (u64, u64, u64),
}

/// A broken invariant is an error line, once per process: it means a read went somewhere
/// wrong, which no amount of hit-rate tuning explains away.
#[cfg(feature = "cuda")]
pub(crate) fn report_bad(bad: (u64, u64, u64)) {
    if bad.0 == 0 {
        return;
    }
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    tracing::error!(
        kind = bad.0,
        a = bad.1,
        b = bad.2,
        "kv slots: a window invariant broke (1 entry names an unclaimed slot, 2 id outside the pool, 3 mapping past the window, 4 victim took a live slot, 5 a copy did not land, 6 a slot a hit read is not its block's bytes)"
    );
}

/// `LEGION_KV_SLOT_SHADOW=1`: the window is kept, filled and counted, but the attention is
/// handed no map, so every read answers from the authority. One run then says whether a wrong
/// output comes from reading a slot at all.
pub(crate) fn shadow() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("LEGION_KV_SLOT_SHADOW").is_ok_and(|v| v != "0");
        if on {
            tracing::warn!("kv slots: SHADOW mode - the window is kept and filled, nothing reads from it");
        }
        on
    })
}

/// The residency map of the pool whose cache tensor is `base`: what the sparse attention
/// maps a selected block through. Null when there is nothing to map - no window, a pool the
/// window does not cover (the drafter's cache), or shadow mode - and null costs the reader
/// nothing at all.
#[cfg(feature = "cuda")]
pub(crate) fn remap_for(base: u64) -> *const i32 {
    if !ACTIVE.load(std::sync::atomic::Ordering::Relaxed) || shadow() {
        return std::ptr::null();
    }
    // SAFETY: plain C; a null result for an unknown tensor is the answer, not an error.
    unsafe { legion_kv_slots_remap_for(base) }
}

/// The window of the registered state, 0: off (the stats snapshot's question).
pub(crate) fn window() -> usize {
    WINDOW.load(std::sync::atomic::Ordering::Relaxed)
}

static WINDOW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The cache's counters (hits, misses, evictions), summed over the registered devices.
/// Reads the resolve kernel's mirrored copy: plain memory, safe on any thread and never
/// a CUDA synchronization; values lag a replay by design (a display gauge).
#[cfg(feature = "cuda")]
pub(crate) fn read_counters() -> SlotCounters {
    if !ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
        return SlotCounters::default();
    }
    let mut out = [0u64; 11];
    let mut sum = [0u64; 6];
    let mut bad = (0u64, 0u64, 0u64);
    let mut devs: Vec<usize> = pools().lock().unwrap().iter().map(|&(_, d)| d).collect();
    devs.sort_unstable();
    devs.dedup();
    for &dev in devs.iter() {
        // SAFETY: plain C, `out` is 6 u64s.
        if unsafe { legion_kv_slots_read(dev as i32, out.as_mut_ptr()) } == 0 {
            for (a, b) in sum.iter_mut().zip(out.iter()) {
                *a += *b;
            }
            if out[6] != 0 && bad.0 == 0 {
                bad = (out[6], out[7], out[8]);
            }
        }
    }
    SlotCounters {
        hits: sum[0],
        misses: sum[1],
        protected: sum[2],
        evictions: sum[3],
        overflow: sum[4],
        drops: sum[5],
        bad,
    }
}

/// The cache tensors of every pool with a state, with the device that holds it. Keyed by
/// tensor, not by device: `--spec mtp` runs a drafter with its own pool on the same GPU,
/// and any second pool would do the same - gating on the device let the resolve rewrite
/// a pool it knows nothing about, reading its block table through these arrays.
#[cfg(feature = "cuda")]
fn pools() -> &'static std::sync::Mutex<Vec<(u64, usize)>> {
    static R: std::sync::OnceLock<std::sync::Mutex<Vec<(u64, usize)>>> = std::sync::OnceLock::new();
    R.get_or_init(Default::default)
}

/// A pool that deliberately has no state, recorded so an unmatched pool is either silent
/// (expected: the drafter) or loud (a real pool that silently got no cache - which, being
/// a missing node, would otherwise look like nothing at all).
#[cfg(feature = "cuda")]
const NO_STATE: usize = usize::MAX;

/// Whether the cache state on `dev` was built over `base` (one of that pool's cache
/// tensors). Graph-build time only: the answer is baked into the node's parameters.
#[cfg(feature = "cuda")]
pub(crate) fn serves(base: u64, dev: usize) -> bool {
    pools().lock().unwrap().iter().any(|&(b, d)| b == base && d == dev)
}

/// Record a pool's cache tensors as deliberately unwindowed.
#[cfg(feature = "cuda")]
pub(crate) fn exclude(segments: &[Seg]) {
    pools()
        .lock()
        .unwrap()
        .extend(segments.iter().map(|g| (g.base, NO_STATE)));
}

/// Warn once per cache tensor that reaches the hook with no state and no exemption.
#[cfg(feature = "cuda")]
fn unknown(base: u64) {
    use std::collections::HashSet;
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<u64>>> = std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    let mut g = seen.lock().unwrap();
    if g.insert(base) {
        tracing::warn!(
            cache = format!("{base:#x}"),
            "kv slots: a sparse attention reads a cache the window knows nothing about (no node built for it)"
        );
    }
}

/// Build device `device`'s slot state for a pool of `n_blocks` blocks whose first `w` are
/// the reserved window, over `segments` (every tiered attention layer's K and V). Called
/// where the pool's tier is built; a rebuild replaces the state (the ids moved).
#[cfg(feature = "cuda")]
pub(crate) fn register(
    device: usize,
    w: usize,
    n_blocks: usize,
    hbm: usize,
    cell_space: usize,
    segments: &[Seg],
) {
    // SAFETY: plain C; `segments` outlives the call.
    let code = unsafe {
        legion_kv_slots_ensure(
            device as i32,
            w as i64,
            n_blocks as i64,
            hbm as i64,
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
    // (Re)index this pool's tensors. The stale bases go first: a resize moves the pool, and
    // a recycled address must not resolve to a state whose segments it is not part of.
    let mut r = pools().lock().unwrap();
    r.retain(|&(_, d)| d != device);
    r.extend(segments.iter().map(|g| (g.base, device)));
    drop(r);
    // From here on a freed block id loses its slot before it can be handed out again
    // (§4.10.6). Idempotent; every pool's free path then reaches the release queue.
    crate::kv::BlockPool::set_free_observer(release);
    ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
    WINDOW.store(w, std::sync::atomic::Ordering::Relaxed);
}

/// Whether any pool registered a slot state (set once, read on the release path and by no
/// hot loop).
#[cfg(feature = "cuda")]
static ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a slot state exists on any device (the stats poll's gate).
#[cfg(feature = "cuda")]
pub(crate) fn active() -> bool {
    ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// No host tier, no slot state on the CPU (mirrors `node` never building the op there).
#[cfg(not(feature = "cuda"))]
pub(crate) fn active() -> bool {
    false
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn read_counters() -> SlotCounters {
    SlotCounters::default()
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn report_bad(_: (u64, u64, u64)) {}

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
/// (`sparse_paged`): reads the selection rows and positions, fills the window's residency map
/// and pulls the claimed blocks host -> slot. It never writes the block table, so no other
/// reader of that table can be affected by it. `None` (no node, an untouched graph) unless a
/// pool registered a state. A device-only op: on the CPU there is no host tier, no
/// registration, and the body is a pass-through kept for the op contract.
#[cfg(feature = "cuda")]
pub(crate) fn node<'a>(
    ctx: &'a crate::ggml::Context,
    k: crate::ggml::Tensor<'_>,
    sel: crate::ggml::Tensor<'a>,
    table: crate::ggml::Tensor<'a>,
    pos: crate::ggml::Tensor<'a>,
) -> Option<crate::ggml::Tensor<'a>> {
    let dev = k.cuda_device()?;
    // Which pool, not which device: the drafter's cache lives on this GPU too, and its
    // block table is not this state's to rewrite.
    let base = unsafe { (*k.raw()).data as u64 };
    if !serves(base, dev) {
        let known = pools().lock().unwrap().iter().any(|&(b, _)| b == base);
        if !known {
            unknown(base);
        }
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
            sel_t.nb[1] as i64 / 4,
            tab_t.data as *mut i32,
            tab_t.nb[1] as i64 / 4,
            pos_t.data as *const i32,
            rows as i32,
            (rows / n_seqs).max(1) as i32,
            n_seqs as i32,
            p[0] - 1,
            stream,
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// Builds the device's slot state; see [`register`]. `hbm` is the pool's tier boundary:
    /// blocks below it are device memory already and are never paged.
    fn legion_kv_slots_ensure(
        dev: i32,
        w: i64,
        n_blocks: i64,
        hbm: i64,
        cell_space: i64,
        segs: *const Seg,
        n_segs: i32,
        copy_max: i32,
    ) -> i32;
    /// The resolve and the copy, on the attention call's stream. `seqs` is the table's own
    /// row count: the kernel must not derive it from `rows / n_tok`, which a step whose rows
    /// are not an exact multiple of it would read past `pos` and the mark arrays with.
    fn legion_kv_slots_graph(
        sel: *const i32,
        sel_nb1: i64,
        tab: *mut i32,
        tab_nb1: i64,
        pos: *const i32,
        rows: i32,
        n_tok: i32,
        seqs: i32,
        dev: i32,
        stream: *mut std::os::raw::c_void,
    );
    /// The residency map of the pool a cache tensor belongs to, null for one with no window.
    fn legion_kv_slots_remap_for(base: u64) -> *const i32;
    /// A freed block's slot dies.
    fn legion_kv_slots_release(id: i32);
    /// The kernel-mirrored counters (11 u64): hits, misses, protected, evictions, overflow,
    /// drops (the last counted host-side in the ring), then the first broken invariant
    /// `(kind, a, b)` and padding. Reads pinned memory, no CUDA call. 3: no state.
    fn legion_kv_slots_read(dev: i32, ctl: *mut u64) -> i32;
}
