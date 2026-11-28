// src/cuda/kv_slots.cu - the KV slot cache on the GPU (the CUDA body of
// src/model/qwen4/slot_cache.rs; the design: docs/kv-streaming-porting-notes.md §4.10, the
// measured basis: §5.3).
//
// On a pool with a host tier, the pool's first `w` blocks are the slot window (reserved out
// of allocation, so no sequence is ever handed an id that a mapping could name). Before an
// attention launch, `legion_kv_slots_graph` runs the model's resolve as two captured,
// synchronization-free kernels on the step's stream:
//   resolve_kernel - one block, phases with barriers, the model's walk step for step:
//     drain the released ids, protect the step's write spans (refreshing a span block that is
//     resident: the write-through, §4.10.3), walk the selections (hits mark the clock, misses
//     claim once, and a block below the tier boundary is never claimed - paging bytes that
//     never crossed the link buys nothing), then the second-chance clock picks victims that
//     take no slot this step reads.
//   copy_kernel - fixed grid, exits on the device count: each entry copies its block's K and
//     V rows authority -> slot, per segment (the registered pool bases). Same tiered virtual
//     range on both sides, so a copy is a plain device copy. It then checks the sampled hits
//     against their authority and retires any mapping that no longer matches.
//
// `residency` (block -> slot, -1: read the authority) is the only truth, and it is read by
// the attention itself. **The block table is never written**, so no other reader of it -
// another layer, another pool, a future path - can be affected by this one. That also means
// a slot taken from a block needs no fixing anywhere else: `residency` clears in the same
// breath, and the next reader falls back to the authority.
//
// The state is per pool, reached by the base address of its cache tensors: a device holds more
// than one pool (the MTP head's cache lives on the last stage's device), and reading the
// target's map for the drafter's blocks would answer "resident" for slots that never held
// them. It is built at pool-build time and read by every later launch and capture.
// The release ring is pinned: the engine writes freed ids between steps (one writer: the
// worker that owns the pool), and every resolve drains it before walking.

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>

namespace {

/// One (kind, layer) copy segment: the pool tensor's base and one block's bytes in it.
struct Seg {
    uintptr_t base;
    long long bytes;
};

/// The pinned release ring plus its cursors (host produces, the kernel consumes; the host
/// reads `consumed` only to watch for drops).
struct ReleaseRing {
    volatile unsigned long long produce;
    volatile unsigned long long consumed;
    volatile unsigned long long ctl_mirror[6]; // = N_CTL, mirrored by the resolve kernel
    volatile unsigned long long drops;         // host-side: a full ring forced a flush
    volatile int flush_all;                    // set by the producer, drained by the resolve
    // The first broken invariant, with the ids, for the reporter to print once. 1: an entry
    // in a selection names a slot nobody claimed (a foreign state rewrote this table). 2: an
    // entry resolves outside the pool. 3: a resident mapping points past the window. 4: a
    // victim took a slot the walk had already read this call. 5: a copy did not land. 6: a
    // slot a hit read does not hold its block's bytes - the mapping went stale somewhere.
    volatile int bad_kind;
    volatile long long bad[4];
    static const int R = 1 << 16;
    volatile int ids[R];
};

/// The ctl counters.
enum { C_HITS = 0, C_MISSES = 1, C_PROTECTED = 2, C_EVICTS = 3, C_OVERFLOW = 4, C_DROPS = 5 };
constexpr int N_CTL = 6;

struct Slots {
    int dev = -1;
    long long w = 0, n_blocks = 0;
    long long hbm = 0;        // blocks resident in the device range: paging one buys nothing
    int32_t* residency = nullptr;   // block -> slot, -1: not resident
    uint32_t* slot_block = nullptr; // slot -> block + 1, 0: never claimed (kept past eviction)
    uint8_t* slot_ref = nullptr;    // the clock's second-chance bit
    unsigned long long* ctl = nullptr;
    int32_t* claim_list = nullptr;  // (copy_max) this call's claims (block ids)
    int2* copy_list = nullptr;      // (copy_max) (block, slot) per claimed copy
    int32_t* copy_count = nullptr;  // device count the copy kernel reads
    int32_t* refresh_count = nullptr; // the write-through copies, queued first, never dropped
    int32_t* protected_mark = nullptr; // (n_blocks) the step's write spans (int: atomicExch)
    int32_t* claimed_mark = nullptr;   // (n_blocks) claim dedup (int: atomicExch)
    int32_t* span_list = nullptr;      // (copy_max) this call's write-span blocks
    int32_t* span_count = nullptr;
    int3* verify_list = nullptr;       // (copy_max) sampled hits: (block, slot, segment)
    int32_t* verify_count = nullptr;
    int32_t* ref_epoch = nullptr;      // (w) the resolve epoch that last read each slot
    int32_t* meta = nullptr;           // [0] the clock hand, [1] the epoch: device-persistent
    Seg* segs = nullptr;               // (n_segs) the copy targets
    int n_segs = 0;
    int copy_max = 0;          // the claim list's cap (the caller bounds it per graph)
    long long copy_stride = 0; // bytes one copy CTA moves per segment
    int copy_chunks = 0;       // grid.x of the copy kernel
    // The clock hand lives in `meta` on the device: `Slots` goes into the kernel by value,
    // so a host-side hand would restart every call at slot 0 and the clock would degrade to
    // "always scan from the front".
    ReleaseRing* ring = nullptr; // pinned, host-visible
};

/// Record the first broken invariant, with the numbers behind it.
__device__ __forceinline__ void bad(Slots& s, int kind, long long a, long long b) {
    if (s.ring->bad_kind == 0) {
        s.ring->bad[0] = a;
        s.ring->bad[1] = b;
        s.ring->bad_kind = kind;
    }
}

/// Drop a pool's device state. `register` runs again on every `resize_kv` (the ids moved),
/// so a rebuild is not a rare path: everything the build allocated goes here.
void free_state(Slots& s) {
    cudaFree(s.residency);   cudaFree(s.slot_block); cudaFree(s.slot_ref);
    cudaFree(s.ctl);         cudaFree(s.claim_list); cudaFree(s.copy_list);
    cudaFree(s.copy_count);  cudaFree(s.refresh_count); cudaFree(s.protected_mark);
    cudaFree(s.claimed_mark); cudaFree(s.ref_epoch);
    cudaFree(s.span_list); cudaFree(s.span_count);
    cudaFree(s.verify_list); cudaFree(s.verify_count);
    cudaFree(s.meta);        cudaFree(s.segs);
    if (s.ring) cudaFreeHost(s.ring);
    s = Slots{};
}

std::map<int, Slots> g_slots;

/// Every cache tensor of every pool with a state, to the state that covers it. Keyed by
/// tensor because a GPU holds more than one pool: the MTP head's cache lives on the last
/// stage's device, and reading the target's remap for its blocks would answer "resident" for
/// a slot that never held them.
std::map<uintptr_t, Slots*> g_by_base;

/// The lock over the two registries. `ensure` inserts (on the thread building graphs, and on
/// every `resize_kv`), `release` runs on the allocator's thread, `read` on the reporter's,
/// `graph` on a worker's: a map insert beside those finds is undefined, not merely racy.
///
/// It guards the containers, not the lifetimes: `graph` looks the state up under it and then
/// launches outside, so a rebuild still has to wait for nothing to be executing (what
/// `resize_kv` does by dropping the graphs first). That part is not solvable here.
std::mutex g_mu;

constexpr int RT = 1024;
constexpr long long COPY_STRIDE = 64 * 1024;

__global__ void resolve_kernel(int32_t* __restrict__ sel, long long sel_nb1, int32_t* __restrict__ tab,
                               long long tab_nb1, const int* __restrict__ pos, int rows, int n_tok,
                               int seqs, Slots s) {
    const int tid = threadIdx.x;
    const int nt = max(n_tok, 1);
    const int n_seqs = max(seqs, 1);
    __shared__ int sh_epoch;
    __shared__ int sh_flush;
    __shared__ int sh_claims;   // step 3's claim count: copy_count ends up including refreshes

    // 0. Drain the released ids (§4.10.6) and open this call's epoch. One thread; releases
    // are rare and few. The epoch is what makes "a slot this step reads" cheap to stamp:
    // the layers that call this need no array cleared between them (§4.10.5).
    if (tid == 0) {
        sh_flush = s.ring->flush_all;
        unsigned long long produce = s.ring->produce;
        while (s.ring->consumed < produce) {
            int id = s.ring->ids[(s.ring->consumed++) % ReleaseRing::R];
            if (id >= 0 && id < s.n_blocks) s.residency[id] = -1;  // another pool's id: skip
        }
        if (sh_flush) s.ring->consumed = produce;  // what never got queued: covered below
        *s.refresh_count = 0;
        *s.span_count = 0;
        *s.verify_count = 0;
        sh_epoch = ++s.meta[1];
    }
    __syncthreads();
    if (sh_flush) {
        // The window goes entirely: an id we could not queue may already have been handed
        // out again, and a slot for it would answer with the previous owner's bytes.
        for (long long i = tid; i < s.n_blocks; i += RT) s.residency[i] = -1;
        if (tid == 0) s.ring->flush_all = 0;
    }
    __syncthreads();
    const int epoch = sh_epoch;

    // 1. Protect the blocks this step's tokens write (§4.10.3): never a claim. The span's
    // table entries are identity ids by that same rule, so no indirection is needed. A span
    // block already resident has its slot refreshed - the write-through (G4) that keeps a
    // resident copy from going stale, queued at the front of the copy list and never dropped
    // for a claim: an entry re-pointed at a slot the step still has to write would read the
    // step's own old bytes. Where the list cannot hold it, the copy is dropped and the block
    // stops being a slot, so the entry stays an identity read.
    for (int i = tid; i < n_seqs * nt; i += RT) {
        int p = pos[i / nt] + i % nt;
        if (p < 0 || p / 64 >= tab_nb1) continue;
        long long cell = (long long)(i / nt) * tab_nb1 + p / 64;
        long long b = tab[cell];  // a block id: the table is never rewritten
        if (b < 0 || b >= s.n_blocks) continue;
        if (atomicExch(&s.protected_mark[b], 1) != 0) continue;  // the first touch records it
        int at = atomicAdd(s.span_count, 1);
        if (at < s.copy_max) s.span_list[at] = (int32_t)b;
        int r = s.residency[b];
        if (r < 0) continue;
        s.ref_epoch[r] = epoch;  // read by this step's attention: off-limits to victims
        int q = atomicAdd(s.refresh_count, 1);
        if (q < s.copy_max) s.copy_list[q] = make_int2((int)b, r);
        else s.residency[b] = -1;  // cannot refresh: drop the copy rather than read it
    }
    __syncthreads();

    // 2. Walk the selections: mark hits, claim each miss's block once, and mark the table
    // cells the walk names for re-pointing.
    if (tid == 0) *s.copy_count = 0;
    __syncthreads();
    const int cap = (int)sel_nb1 - 1;
    for (int i = tid; i < rows * cap; i += RT) {
        int row = i / cap, slot = i % cap;
        const int32_t* r = sel + (long long)row * sel_nb1;
        if (slot >= (r[0] < 0 ? 0 : r[0])) continue;
        int p = r[1 + slot];
        if (p < 0) continue;
        // Same-block cells concentrate: positions ascend, the previous cell says it.
        if (slot > 0 && r[slot] / 64 == p / 64) continue;
        int seq = min(row / nt, n_seqs - 1);
        long long cell = (long long)seq * tab_nb1 + p / 64;
        // The table is never rewritten: an entry is a block id, and the reader maps it
        // through `residency` itself. So there is nothing to resolve here but the block.
        long long b = tab[cell];
        if (b < 0) continue;  // a padded row
        if (b >= s.n_blocks) {
            bad(s, 2, b, cell);
            continue;
        }
        int32_t v = s.residency[b];
        if (v >= s.w) bad(s, 3, b, v);
        if (s.protected_mark[b]) {
            // A block this step writes: its slot, if it has one, was refreshed above, and the
            // attention reads it through the remap either way.
            if (v < 0) atomicAdd(&s.ctl[C_PROTECTED], 1ULL);
            continue;
        }
        if (v >= 0) {
            // Served from the slot. It needs no protection from the victim sweep: were the
            // slot taken below, `residency[b]` would clear in the same breath and every later
            // reader would fall back to the authority.
            s.slot_ref[v] = 1;
            atomicAdd(&s.ctl[C_HITS], 1ULL);
            // Sample a fraction of the hits for the check in the copy kernel. This is the
            // one property the whole design rests on - a slot's bytes are its block's bytes
            // - and a stale slot is a wrong read that no counter shows. The hash spreads the
            // samples across blocks and across steps, so a write that fails to refresh is
            // caught within a few hundred reads of happening.
            unsigned h = ((unsigned)b * 2654435761u) ^ ((unsigned)epoch * 40503u);
            if ((h & 1023u) == 0u) {
                int at = atomicAdd(s.verify_count, 1);
                // One segment per check, rotated by the same hash. A block has rows in every
                // layer's K and V; comparing all of them would put a megabyte-scale compare
                // on a single CTA - the very cost this feature exists to avoid paying. The
                // rotation covers every segment across steps instead.
                if (at < s.copy_max)
                    s.verify_list[at] = make_int3((int)b, (int)v,
                                                  (int)((h >> 10) % (unsigned)s.n_segs));
            }
            continue;
        }
        // A block below the tier boundary is device memory already: claiming it spends a
        // window slot and a full copy on bytes that never crossed the link, and evicts a
        // host block that would have.
        if (b < s.hbm) continue;
        if (atomicExch(&s.claimed_mark[b], 1) == 0) {
            int at = atomicAdd(s.copy_count, 1);
            if (at < s.copy_max)
                s.claim_list[at] = (int32_t)b;
            else {
                s.claimed_mark[b] = 0; // the list is full: read it as identity this step
                atomicAdd(&s.ctl[C_OVERFLOW], 1ULL);
            }
        }
    }
    __syncthreads();

    // 3. Victims and re-pointing targets, one thread's clock (the model's step 4): never a
    // slot the live table points at, never one this call handed out. The walk's claims are
    // few (a warm step has none), and the first cold step pays one thread over the window.
    if (tid == 0) {
        int n_claims = min(*s.copy_count, s.copy_max);
        int nr = min(*s.refresh_count, s.copy_max);
        // Nothing to clear and no table to scan: the walk and the refresh already stamped
        // every slot this step reads with the epoch. The hand is the device's, so the clock
        // actually moves between calls.
        long long hand = s.meta[0];
        int out = nr;
        for (int i = 0; i < n_claims; i++) {
            long long b = s.claim_list[i];
            long long taken = -1;
            for (int tries = 0; tries < 2 * (int)s.w; tries++) {
                long long v = hand;
                hand = (hand + 1) % max((int)s.w, 1);
                if (s.slot_ref[v]) {
                    s.slot_ref[v] = 0;
                    continue;
                }
                // Held back on purpose: the walk, the refresh or the live table reads it.
                if (s.ref_epoch[v] == epoch) continue;
                unsigned long long old = s.slot_block[v];
                if (old != 0 && s.residency[old - 1] == (int32_t)v) {
                    s.residency[old - 1] = -1;
                    s.ctl[C_EVICTS]++;
                }
                taken = v;
                break;
            }
            if (taken < 0 || out >= s.copy_max) {
                s.claimed_mark[b] = 0;  // give the mark back: else never cached again
                atomicAdd(&s.ctl[C_OVERFLOW], 1ULL);
                continue;
            }
            s.residency[b] = (int32_t)taken;
            s.slot_block[taken] = (uint32_t)(b + 1);
            s.slot_ref[taken] = 1;
            s.ref_epoch[taken] = epoch;
            s.copy_list[out++] = make_int2((int)b, (int)taken);
            atomicAdd(&s.ctl[C_MISSES], 1ULL);
        }
        s.meta[0] = (int32_t)hand;
        sh_claims = n_claims;
        *s.copy_count = out;
    }
    __syncthreads();

    // 4. Clear this step's marks. Nothing needs undoing in the table - this kernel never
    // writes it; the readers map blocks through `residency`, which is already correct.
    // The marks go by the ids recorded above, not by reading anything back: this kernel
    // writes no table, and a slot's owner may have changed underneath it.
    for (int i = tid, n = min(*s.span_count, s.copy_max); i < n; i++) {
        int b = s.span_list[i];
        if (b >= 0 && b < s.n_blocks) s.protected_mark[b] = 0;
    }
    for (int i = tid, n = sh_claims; i < n; i++) {
        int b = s.claim_list[i];
        if (b >= 0 && b < s.n_blocks) s.claimed_mark[b] = 0;
    }
    // Mirror the counters for the host's stats poll (see ReleaseRing::ctl_mirror).
    if (tid == 0)
        for (int i = 0; i < N_CTL; i++) s.ring->ctl_mirror[i] = s.ctl[i];
}

/// Bytes compared at each end of one sampled (block, slot) pair, within one segment. A slot
/// that went stale because a write never reached it differs at its tail - the cells a step
/// writes are the newest ones, and at most a step's worth of them land in one block - while a
/// slot answering for the wrong block differs everywhere. Both ends therefore cover the
/// practical cases; a block smaller than the two ends is compared whole (§seg_differs).
constexpr long long CHECK_ENDS = 16 * 1024;

/// Whether this thread's slice of one segment differs between block `a` and slot `b`.
__device__ __forceinline__ bool seg_differs(const Seg& sg, long long a, long long b) {
    const unsigned char* pa = (const unsigned char*)sg.base + a * sg.bytes;
    const unsigned char* pb = (const unsigned char*)sg.base + b * sg.bytes;
    const long long head = min(CHECK_ENDS, sg.bytes);
    const long long tail = min(CHECK_ENDS, sg.bytes - head);  // never the same bytes twice
    for (long long i = (long long)threadIdx.x * 16; i < head + tail;
         i += (long long)blockDim.x * 16) {
        const long long o = i < head ? i : sg.bytes - tail + (i - head);
        const uint4 x = *(const uint4*)(pa + o);
        const uint4 y = *(const uint4*)(pb + o);
        if (x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w) return true;
    }
    return false;
}

/// One CTA moves `copy_stride` bytes of one claimed block across all segments (a loop: the
/// segment count is a few dozen at most), then checks the sampled hits. Exits on the device
/// counts, so the fixed grid is free on a warm step.
__global__ void copy_kernel(Slots s) {
    const int c = blockIdx.y;
    const int nc = *s.copy_count;
    if (c < nc) {
        int2 w = s.copy_list[c];
        long long off = (long long)blockIdx.x * s.copy_stride;
        for (int g = 0; g < s.n_segs; g++) {
            const Seg& sg = s.segs[g];
            if (off >= sg.bytes) continue;
            long long n = min(s.copy_stride, sg.bytes - off);
            const uint4* src = (const uint4*)(sg.base + (long long)w.x * sg.bytes + off);
            uint4* dst = (uint4*)(sg.base + (long long)w.y * sg.bytes + off);
            for (long long i = threadIdx.x; i * 16 < n; i += blockDim.x) {
                const uint4 v = src[i];
                dst[i] = v;
                // Only that the write landed: reading back what was just stored cannot show
                // a wrong source. The check of the mapping itself is below, where a hit is
                // compared with its authority.
                uint4 back = dst[i];
                if (back.x != v.x || back.y != v.y || back.z != v.z || back.w != v.w) {
                    if (s.ring->bad_kind == 0) {
                        s.ring->bad[0] = w.x;
                        s.ring->bad[1] = w.y;
                        s.ring->bad_kind = 5;
                    }
                }
            }
        }
        return;
    }
    // The check a hit asked for: does the slot still hold its block's bytes? One CTA per
    // pair, on the one segment that hit chose. If it does not, the mapping is retired on the
    // spot - every later read of that block falls back to the authority - and the first such
    // pair is reported. So a stale slot costs one wrong read at most, and the run says
    // whether it happened at all.
    const int vi = c - nc;
    if (vi >= *s.verify_count || blockIdx.x != 0) return;
    int3 p = s.verify_list[vi];
    if (p.z < 0 || p.z >= s.n_segs) return;
    __shared__ int sh_diff;
    if (threadIdx.x == 0) sh_diff = 0;
    __syncthreads();
    if (seg_differs(s.segs[p.z], p.x, p.y)) atomicOr(&sh_diff, 1);
    __syncthreads();
    if (sh_diff != 0) {
        s.residency[p.x] = -1;
        bad(s, 6, p.x, p.y);
    }
}

}  // namespace

extern "C" {

/// Builds device `dev`'s slot state over a pool of `n_blocks` with a `w`-block window, the
/// copy segments (`segs`, `n_segs`: every tiered attention layer's K and V) and the
/// claim-list cap. A CUDA error code, 0 for good.
int legion_kv_slots_ensure(int dev, long long w, long long n_blocks, long long hbm,
                           long long cell_space, const Seg* segs, int n_segs, int copy_max) {
    // The copy and the hit check both move `uint4`: a block whose bytes are not a whole
    // number of vectors, a base that is not vector-aligned, or a window wider than the pool
    // would read out of step rather than fail. Refuse the build; the caller logs it.
    if (w <= 0 || n_blocks <= 0 || w > n_blocks || n_segs <= 0 || copy_max <= 0) return 4;
    for (int i = 0; i < n_segs; i++)
        if (segs[i].base % 16 || segs[i].bytes % 16) return 4;
    Slots s;
    s.dev = dev;
    s.w = w;
    s.n_blocks = n_blocks;
    s.hbm = hbm;
    s.n_segs = n_segs;
    s.copy_max = copy_max;
    long long max_bytes = 1;
    for (int i = 0; i < n_segs; i++)
        if (segs[i].bytes > max_bytes) max_bytes = segs[i].bytes;
    s.copy_stride = COPY_STRIDE;
    s.copy_chunks = (int)((max_bytes + COPY_STRIDE - 1) / COPY_STRIDE);
    int prev;
    cudaGetDevice(&prev);
    cudaSetDevice(dev);
    bool ok = cudaMalloc((void**)&s.residency, n_blocks * 4) == cudaSuccess &&
              cudaMalloc((void**)&s.slot_block, w * 4) == cudaSuccess &&
              cudaMalloc((void**)&s.slot_ref, w) == cudaSuccess &&
              cudaMalloc((void**)&s.ctl, N_CTL * 8) == cudaSuccess &&
              cudaMalloc((void**)&s.claim_list, copy_max * 4) == cudaSuccess &&
              cudaMalloc((void**)&s.copy_list, copy_max * sizeof(int2)) == cudaSuccess &&
              cudaMalloc((void**)&s.copy_count, 4) == cudaSuccess &&
              cudaMalloc((void**)&s.refresh_count, 4) == cudaSuccess &&
              cudaMalloc((void**)&s.protected_mark, n_blocks * 4) == cudaSuccess &&
              cudaMalloc((void**)&s.claimed_mark, n_blocks * 4) == cudaSuccess &&
              cudaMalloc((void**)&s.span_list, copy_max * 4) == cudaSuccess &&
              cudaMalloc((void**)&s.span_count, 4) == cudaSuccess &&
              cudaMalloc((void**)&s.verify_list, copy_max * sizeof(int3)) == cudaSuccess &&
              cudaMalloc((void**)&s.verify_count, 4) == cudaSuccess &&
              cudaMalloc((void**)&s.ref_epoch, w * 4) == cudaSuccess &&
              cudaMalloc((void**)&s.meta, 8) == cudaSuccess &&
              cudaMalloc((void**)&s.segs, n_segs * sizeof(Seg)) == cudaSuccess &&
              cudaHostAlloc((void**)&s.ring, sizeof(ReleaseRing), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaMemset(s.residency, 0xff, n_blocks * 4) == cudaSuccess &&
         cudaMemset(s.slot_block, 0, w * 4) == cudaSuccess &&
         cudaMemset(s.slot_ref, 0, w) == cudaSuccess &&
         cudaMemset(s.ctl, 0, N_CTL * 8) == cudaSuccess &&
         cudaMemset(s.copy_count, 0, 4) == cudaSuccess &&
         cudaMemset(s.refresh_count, 0, 4) == cudaSuccess &&
         cudaMemset(s.protected_mark, 0, n_blocks * 4) == cudaSuccess &&
         cudaMemset(s.claimed_mark, 0, n_blocks * 4) == cudaSuccess &&
         cudaMemset(s.span_count, 0, 4) == cudaSuccess &&
         cudaMemset(s.verify_count, 0, 4) == cudaSuccess &&
         cudaMemset(s.ref_epoch, 0, w * 4) == cudaSuccess &&
         cudaMemset(s.meta, 0, 8) == cudaSuccess;
    if (ok) {
        s.ring->produce = 0;
        s.ring->consumed = 0;
        cudaMemcpy(s.segs, segs, n_segs * sizeof(Seg), cudaMemcpyHostToDevice);
        std::lock_guard<std::mutex> lock(g_mu);
        Slots& into = g_slots[dev];
        // Drop the previous pool's entries by owner, not by re-reading its segments: `segs`
        // in the state is the device copy, and indexing it here would fault on the host.
        for (auto b = g_by_base.begin(); b != g_by_base.end();)
            b = (b->second == &into) ? g_by_base.erase(b) : std::next(b);
        if (into.residency) free_state(into);  // a rebuild: the ids moved
        into = s;
        // Index from the caller's host array - the only copy of the bases readable here.
        for (int i = 0; i < n_segs; i++) g_by_base[(uintptr_t)segs[i].base] = &into;
    } else {
        free_state(s);  // a partial build: nothing is registered, so nothing holds these
    }
    cudaSetDevice(prev);
    return ok ? 0 : 1;
}

/// The engine's release hook: a freed block's slot dies (the id may be re-used). The one
/// writer is the worker that owns the pool; a full ring (a stalled drain over 64K frees)
/// drops and counts - the walk would then read a stale slot for a re-used id, which the
/// count makes visible where it matters.
void legion_kv_slots_release(int id) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto& kv : g_slots) {
        ReleaseRing* r = kv.second.ring;
        unsigned long long p = r->produce;
        if (p - r->consumed >= (unsigned long long)ReleaseRing::R) {
            r->drops++;  // pinned: `ctl` is device memory, and a full ring is not a crash
            r->flush_all = 1;  // flush the window rather than lose the release
            continue;
        }
        r->ids[p % ReleaseRing::R] = id;
        __atomic_store_n(&r->produce, p + 1, __ATOMIC_RELEASE);
    }
}

/// The captured-graph hook: resolve, then the copy, on `stream`. Called from the
/// `qsa_kv_slots` op with the attention node's own operands.
void legion_kv_slots_graph(const int* sel, long long sel_nb1, int* tab, long long tab_nb1,
                           const int* pos, int rows, int n_tok, int seqs, int dev,
                           void* stream) {
    Slots* state = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_slots.find(dev);
        if (it != g_slots.end()) state = &it->second;
    }
    if (!state) return;
    Slots& s = *state;
    resolve_kernel<<<1, RT, 0, (cudaStream_t)stream>>>(
        (int32_t*)sel, sel_nb1, tab, tab_nb1, pos, rows, n_tok, seqs, s);
    dim3 grid(s.copy_chunks, s.copy_max);
    copy_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(s);
}

/// The residency map of the pool whose cache tensor is `base` - what a reader maps a block
/// id through. Null when that pool has no window (and for the drafter's cache), which is the
/// reader's "off": no branch, no extra load.
const int* legion_kv_slots_remap_for(unsigned long long base) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_by_base.find((uintptr_t)base);
    return it == g_by_base.end() ? nullptr : it->second->residency;
}

/// Copies the counters out (the reporter's readback): N_CTL u64. 3: no state.
int legion_kv_slots_read(int dev, unsigned long long* ctl) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_slots.find(dev);
    if (it == g_slots.end()) return 3;
    // The kernel-mirrored copy: plain memory, no CUDA sync (safe on any thread).
    const volatile unsigned long long* m = it->second.ring->ctl_mirror;
    for (int i = 0; i < N_CTL; i++) ctl[i] = m[i];
    ctl[C_DROPS] = it->second.ring->drops;  // counted host-side, so read it from the ring
    const volatile ReleaseRing* r = it->second.ring;
    ctl[6] = (unsigned long long)r->bad_kind;  // the first broken invariant, if any
    for (int i = 0; i < 4; i++) ctl[7 + i] = (unsigned long long)r->bad[i];
    return 0;
}

}  // extern "C"
