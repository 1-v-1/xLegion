// src/cuda/kv_slots.cu - the KV slot cache on the GPU (the CUDA body of
// src/model/qwen4/slot_cache.rs; the design: docs/kv-streaming-porting-notes.md §4.10, the
// measured basis: §5.3).
//
// On a pool with a host tier, the pool's first `w` blocks are the slot window (reserved out
// of allocation, so every sequence's block is a host-backed identity id). Before an
// attention launch, `legion_kv_slots_graph` runs the model's resolve as two captured,
// synchronization-free kernels on the step's stream:
//   resolve_kernel - one block, phases with barriers, the model's walk step for step:
//     drain the released ids, protect the step's write spans, walk the selections (hits mark
//     the clock, misses claim once), the second-chance clock picks victims that never take a
//     slot the live table or this call needs, every walked table cell is re-pointed to its
//     block's slot, and the copy list (block, slot) is left in device memory.
//   copy_kernel - fixed grid, exits on the device count: each entry copies its block's K and
//     V rows host-identity -> slot, per segment (the registered pool bases). Same tiered
//     virtual range on both sides, so a copy is a plain device copy; a block this step
//     writes is never a claim, so a slot's bytes never go stale (§4.10.3).
//
// The state is per device (the registry pattern of cvec.cu), built at pool-build time and
// read by every later launch and capture: the graphs bake no address the load could move.
// The release ring is pinned: the engine writes freed ids between steps (one writer: the
// worker that owns the pool), and every resolve drains it before walking.

#include <cuda_runtime.h>

#include <cstdint>
#include <map>

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
    static const int R = 1 << 16;
    volatile int ids[R];
};

/// The ctl counters.
enum { C_HITS = 0, C_MISSES = 1, C_PROTECTED = 2, C_EVICTS = 3, C_OVERFLOW = 4, C_DROPS = 5 };
constexpr int N_CTL = 6;

struct Slots {
    int dev = -1;
    long long w = 0, n_blocks = 0;
    int32_t* residency = nullptr;   // block -> slot, -1: not resident
    uint32_t* slot_block = nullptr; // slot -> block + 1, 0: never claimed (kept past eviction)
    uint8_t* slot_ref = nullptr;    // the clock's second-chance bit
    unsigned long long* ctl = nullptr;
    int32_t* claim_list = nullptr;  // (copy_max) this call's claims (block ids)
    int2* copy_list = nullptr;      // (copy_max) (block, slot) per claimed copy
    int32_t* copy_count = nullptr;  // device count the copy kernel reads
    int32_t* refresh_count = nullptr; // the write-through copies, queued first, never dropped
    int32_t* protected_mark = nullptr; // (n_blocks) the step's write spans (int: atomicExch)
    uint8_t* claimed_mark = nullptr;   // (n_blocks) claim dedup
    uint8_t* requested = nullptr;      // (cell_space) table cells to re-point
    uint8_t* referenced = nullptr;     // (w) slots the live table points at
    Seg* segs = nullptr;               // (n_segs) the copy targets
    int n_segs = 0;
    int copy_max = 0;          // the claim list's cap (the caller bounds it per graph)
    long long copy_stride = 0; // bytes one copy CTA moves per segment
    int copy_chunks = 0;       // grid.x of the copy kernel
    long long hand = 0;        // the clock hand (the resolve kernel is one block)
    ReleaseRing* ring = nullptr; // pinned, host-visible
};

std::map<int, Slots> g_slots;

constexpr int RT = 1024;
constexpr long long COPY_STRIDE = 64 * 1024;

/// The model's `entry_block`: at-or-above `w` the entry is its own block; below, the slot's
/// `slot_block` names whose bytes it holds. A never-claimed slot (`0`) maps to nothing.
__device__ __forceinline__ long long entry_block(int entry, long long w, const uint32_t* slot_block) {
    if (entry >= w) return entry;
    unsigned long long b = slot_block[entry];
    return b == 0 ? -1 : (long long)(b - 1);
}

__global__ void resolve_kernel(int32_t* __restrict__ sel, long long sel_nb1, int32_t* __restrict__ tab,
                               long long tab_nb1, const int* __restrict__ pos, int rows, int n_tok,
                               Slots s) {
    const int tid = threadIdx.x;
    const int nt = max(n_tok, 1);
    const int n_seqs = max(rows / nt, 1);

    // 0. Drain the released ids (§4.10.6). One thread: releases are rare and few.
    if (tid == 0) {
        unsigned long long produce = s.ring->produce;
        while (s.ring->consumed < produce)
            s.residency[s.ring->ids[(s.ring->consumed++) % ReleaseRing::R]] = -1;
    }

    // 1. Protect the blocks this step's tokens write (§4.10.3): never a claim. The span's
    // table entries are identity ids by that same rule, so no indirection is needed. A span
    // block already resident has its slot refreshed - the write-through (G4) that keeps a
    // resident copy from going stale, queued at the front of the copy list and never dropped
    // for a claim: an entry re-pointed at a slot the step still has to write would read the
    // step's own old bytes. Where the list cannot hold it, the copy is dropped and the block
    // stops being a slot, so the entry stays an identity read.
    if (tid == 0) *s.refresh_count = 0;
    __syncthreads();
    for (int i = tid; i < n_seqs * nt; i += RT) {
        int p = pos[i / nt] + i % nt;
        long long cell = (long long)(i / nt) * tab_nb1 + p / 64;
        int e = tab[cell];
        if (e < 0 || e >= s.n_blocks) continue;
        if (atomicExch(&s.protected_mark[e], 1) != 0) continue;  // the first touch queues it
        int r = s.residency[e];
        if (r < 0) continue;
        int at = atomicAdd(s.refresh_count, 1);
        if (at < s.copy_max) s.copy_list[at] = make_int2(e, r);
        else s.residency[e] = -1;
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
        long long b = entry_block(tab[cell], s.w, s.slot_block);
        if (b < 0 || b >= s.n_blocks) continue;
        int32_t v = s.residency[b];
        if (s.protected_mark[b]) {
            if (v < 0) {
                atomicAdd(&s.ctl[C_PROTECTED], 1ULL);
                continue;
            }
            s.requested[cell] = 1;  // refreshed above: the slot holds the bytes to read
            s.slot_ref[v] = 1;
            atomicAdd(&s.ctl[C_HITS], 1ULL);
            continue;
        }
        s.requested[cell] = 1;
        if (v >= 0) {
            s.slot_ref[v] = 1;
            atomicAdd(&s.ctl[C_HITS], 1ULL);
            continue;
        }
        if (!s.claimed_mark[b]) {
            s.claimed_mark[b] = 1;
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
        for (long long v = 0; v < s.w; v++) s.referenced[v] = 0;
        for (long long c = 0; c < (long long)n_seqs * tab_nb1; c++) {
            int e = tab[c];
            if (e >= 0 && e < s.w) s.referenced[e] = 1;
        }
        // A refreshed slot is read by this step's attention: off-limits to victims.
        for (int i = 0; i < nr; i++) s.referenced[s.copy_list[i].y] = 1;
        int out = nr;
        for (int i = 0; i < n_claims; i++) {
            long long b = s.claim_list[i];
            long long taken = -1;
            for (int tries = 0; tries < 2 * (int)s.w; tries++) {
                long long v = s.hand;
                s.hand = (s.hand + 1) % max((int)s.w, 1);
                if (s.slot_ref[v]) {
                    s.slot_ref[v] = 0;
                    continue;
                }
                if (s.referenced[v]) continue;
                unsigned long long old = s.slot_block[v];
                if (old != 0 && s.residency[old - 1] == (int32_t)v) {
                    s.residency[old - 1] = -1;
                    s.ctl[C_EVICTS]++;
                }
                taken = v;
                break;
            }
            if (taken < 0 || out >= s.copy_max) {
                atomicAdd(&s.ctl[C_OVERFLOW], 1ULL);
                continue;
            }
            s.residency[b] = (int32_t)taken;
            s.slot_block[taken] = (uint32_t)(b + 1);
            s.slot_ref[taken] = 1;
            s.referenced[taken] = 1;
            s.copy_list[out++] = make_int2((int)b, (int)taken);
            atomicAdd(&s.ctl[C_MISSES], 1ULL);
        }
        *s.copy_count = out;
    }
    __syncthreads();

    // 4. Re-point every walked cell to its block's slot, then clear this step's marks. The
    // write spans were never requested, so their entries still hold the ids marked above.
    for (long long c = tid; c < (long long)n_seqs * tab_nb1; c += RT) {
        if (!s.requested[c]) continue;
        long long b = entry_block(tab[c], s.w, s.slot_block);
        if (b < 0) continue;
        int32_t v = s.residency[b];
        if (v >= 0) tab[c] = v;
        s.requested[c] = 0;
    }
    for (int i = tid; i < n_seqs * nt; i += RT) {
        int p = pos[i / nt] + i % nt;
        // The entry may now point at the slot this block was refreshed into; `slot_block`
        // names the block either way.
        long long e = entry_block(tab[(long long)(i / nt) * tab_nb1 + p / 64], s.w, s.slot_block);
        if (e >= 0 && e < s.n_blocks) s.protected_mark[e] = 0;
    }
    int n_claims = min(*s.copy_count, s.copy_max);
    for (int i = tid; i < n_claims; i++) {
        int b = s.copy_list[i].x;
        if (b >= 0 && b < s.n_blocks) s.claimed_mark[b] = 0;
    }
    // Mirror the counters for the host's stats poll (see ReleaseRing::ctl_mirror).
    if (tid == 0)
        for (int i = 0; i < N_CTL; i++) s.ring->ctl_mirror[i] = s.ctl[i];
}

/// One CTA moves `copy_stride` bytes of one claimed block across all segments (a loop: the
/// segment count is a few dozen at most). Exits on the device count, so the fixed grid is
/// free on a warm step.
__global__ void copy_kernel(Slots s) {
    const int c = blockIdx.y;
    if (c >= *s.copy_count) return;
    int2 w = s.copy_list[c];
    long long off = (long long)blockIdx.x * s.copy_stride;
    for (int g = 0; g < s.n_segs; g++) {
        const Seg& sg = s.segs[g];
        if (off >= sg.bytes) continue;
        long long n = min(s.copy_stride, sg.bytes - off);
        const uint4* src = (const uint4*)(sg.base + (long long)w.x * sg.bytes + off);
        uint4* dst = (uint4*)(sg.base + (long long)w.y * sg.bytes + off);
        for (long long i = threadIdx.x; i * 16 < n; i += blockDim.x) dst[i] = src[i];
    }
}

}  // namespace

extern "C" {

/// Builds device `dev`'s slot state over a pool of `n_blocks` with a `w`-block window, the
/// copy segments (`segs`, `n_segs`: every tiered attention layer's K and V) and the
/// claim-list cap. A CUDA error code, 0 for good.
int legion_kv_slots_ensure(int dev, long long w, long long n_blocks, long long cell_space,
                           const Seg* segs, int n_segs, int copy_max) {
    Slots s;
    s.dev = dev;
    s.w = w;
    s.n_blocks = n_blocks;
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
              cudaMalloc((void**)&s.claimed_mark, n_blocks) == cudaSuccess &&
              cudaMalloc((void**)&s.requested, cell_space) == cudaSuccess &&
              cudaMalloc((void**)&s.referenced, w) == cudaSuccess &&
              cudaMalloc((void**)&s.segs, n_segs * sizeof(Seg)) == cudaSuccess &&
              cudaHostAlloc((void**)&s.ring, sizeof(ReleaseRing), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaMemset(s.residency, 0xff, n_blocks * 4) == cudaSuccess &&
         cudaMemset(s.slot_block, 0, w * 4) == cudaSuccess &&
         cudaMemset(s.slot_ref, 0, w) == cudaSuccess &&
         cudaMemset(s.ctl, 0, N_CTL * 8) == cudaSuccess &&
         cudaMemset(s.copy_count, 0, 4) == cudaSuccess &&
         cudaMemset(s.refresh_count, 0, 4) == cudaSuccess &&
         cudaMemset(s.protected_mark, 0, n_blocks * 4) == cudaSuccess &&
         cudaMemset(s.claimed_mark, 0, n_blocks) == cudaSuccess &&
         cudaMemset(s.requested, 0, cell_space) == cudaSuccess;
    if (ok) {
        s.ring->produce = 0;
        s.ring->consumed = 0;
        cudaMemcpy(s.segs, segs, n_segs * sizeof(Seg), cudaMemcpyHostToDevice);
        Slots& into = g_slots[dev];
        if (into.residency) cudaFree(into.residency); // a rebuild: drop the old state
        into = s;
    }
    cudaSetDevice(prev);
    return ok ? 0 : 1;
}

/// The engine's release hook: a freed block's slot dies (the id may be re-used). The one
/// writer is the worker that owns the pool; a full ring (a stalled drain over 64K frees)
/// drops and counts - the walk would then read a stale slot for a re-used id, which the
/// count makes visible where it matters.
void legion_kv_slots_release(int id) {
    for (auto& kv : g_slots) {
        ReleaseRing* r = kv.second.ring;
        unsigned long long p = r->produce;
        if (p - r->consumed >= (unsigned long long)ReleaseRing::R) {
            kv.second.ctl[C_DROPS]++;
            continue;
        }
        r->ids[p % ReleaseRing::R] = id;
        r->produce = p + 1;
    }
}

/// The captured-graph hook: resolve, then the copy, on `stream`. Called from the
/// `qsa_kv_slots` op with the attention node's own operands.
void legion_kv_slots_graph(const int* sel, long long sel_nb1, int* tab, long long tab_nb1,
                           const int* pos, int rows, int n_tok, int dev, void* stream) {
    const auto it = g_slots.find(dev);
    if (it == g_slots.end()) return;
    Slots& s = it->second;
    resolve_kernel<<<1, RT, 0, (cudaStream_t)stream>>>(
        (int32_t*)sel, sel_nb1, tab, tab_nb1, pos, rows, n_tok, s);
    dim3 grid(s.copy_chunks, s.copy_max);
    copy_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(s);
}

/// Copies the counters out (the reporter's readback): N_CTL u64. 3: no state.
int legion_kv_slots_read(int dev, unsigned long long* ctl) {
    const auto it = g_slots.find(dev);
    if (it == g_slots.end()) return 3;
    // The kernel-mirrored copy: plain memory, no CUDA sync (safe on any thread).
    const volatile unsigned long long* m = it->second.ring->ctl_mirror;
    for (int i = 0; i < N_CTL; i++) ctl[i] = m[i];
    return 0;
}

}  // extern "C"
