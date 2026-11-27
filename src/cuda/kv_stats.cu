// src/cuda/kv_stats.cu - the KV slot cache's Phase 0 measurement (the CUDA side of
// src/model/qwen4/kv_stats.rs, docs/kv-streaming-porting-notes.md): count where the QSA
// selections land, per block, on a pool that has a host tier.
//
// The kernel reads the selection rows and the block table, which are graph data already, and
// writes only the counters, so the model's bits are the same with the measurement on or off.
// The counters live in device memory at fixed addresses, one set per device (the registry
// pattern of cvec.cu: `legion_kv_stats_ensure` builds them at pool-build time, and a launch
// only reads the map), so the captured graphs never bake an address the load could move.
// The readback is a synchronous copy the reporter thread runs outside the graphs.
//
// What a step's rows say (per bucket: 0 decode and verify, 8 prompt):
//   ctl[off+0] selected cells in VRAM blocks     (the local reads, today's fast path)
//   ctl[off+1] selected cells in host blocks     (what the link carries per token today)
//   ctl[off+2] host block fetches                (deduplicated per row: what a slot pool
//                                                 would move once and could then keep)
//   ctl[off+3] rows                              (selected cells per row = the selection width)
// and hist[b] counts the fetches of host block b, the concentration a cache of S slots
// could absorb.

#include <cuda_runtime.h>

#include <cstdint>
#include <map>

extern "C" {

/// The kernel's parameters (filled by the Rust launcher from the node's tensors): plain
/// pointers and sizes; the counter pointers are taken from the registry by `dev`.
struct kv_stats_args {
    const int* sel;      // (rows, cap+1) I32: [count, positions ascending, -1] (qsa_attn.cu's form)
    long long sel_nb1;   // elements between rows
    const int* tab;      // (max_blocks, n_seqs) I32 block tables
    long long tab_nb1;
    int cap;             // position slots per row (sel's ne[0] - 1)
    int rows;            // selection rows (the step's tokens)
    int n_tok;           // tokens per sequence (rows / n_seqs)
    long long hbm;       // block ids below this live in VRAM
    long long n_host;    // host blocks (the hist's capacity as of graph build)
    int dev;             // the device the counters belong to
    int off;             // 0: decode and verify, 8: prompt
};

}  // extern "C"

namespace {

struct Ctr {
    unsigned long long* hist = nullptr;  // (n_host,) fetches per host block
    unsigned long long* ctl = nullptr;   // 16 u64
    long long n_host = 0;
};

std::map<int, Ctr> g_ctr;  // one pool per device (the tiers are per device)

__global__ void kv_stats_kernel(const int* __restrict__ sel, long long sel_nb1,
                                const int* __restrict__ tab, long long tab_nb1, int cap, int rows,
                                int n_tok, long long hbm, long long n_host, int off,
                                unsigned long long* hist, unsigned long long* ctl) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * cap;
    if (idx >= total) return;
    const int row = (int)(idx / cap);
    const int slot = (int)(idx % cap);
    const int* r = sel + (long long)row * sel_nb1;
    if (slot == 0) atomicAdd(&ctl[off + 3], 1ULL);
    const int cnt = r[0];
    if (slot >= (cnt < 0 ? 0 : cnt)) return;
    const int pos = r[1 + slot];
    if (pos < 0) return;
    const int blk = tab[(long long)(row / n_tok) * tab_nb1 + pos / 64];
    if (blk < 0) return;
    if ((long long)blk < hbm) {
        atomicAdd(&ctl[off + 0], 1ULL);
        return;
    }
    atomicAdd(&ctl[off + 1], 1ULL);
    const long long h = (long long)blk - hbm;
    if (h >= n_host) return;  // the pool moved since the graph was built: totals only
    // The positions are ascending, so an earlier cell of the same row says this block was
    // already fetched for this token: the histogram counts each fetch once, as a slot pool
    // would move it once.
    if (slot > 0 && r[slot] / 64 == pos / 64) return;
    atomicAdd(&hist[h], 1ULL);
    atomicAdd(&ctl[off + 2], 1ULL);
}

}  // namespace

extern "C" {

/// Builds device `dev`'s counters for a pool with `n_host` host blocks, zeroed. A resize
/// moves the host range's block ids, so a changed `n_host` starts the counters over; a pool
/// rebuilt at the same size keeps them. A CUDA error code, 0 for good.
int legion_kv_stats_ensure(int dev, long long n_host) {
    Ctr& c = g_ctr[dev];
    if (c.hist != nullptr && c.n_host == n_host) return 0;
    if (c.hist != nullptr) {
        cudaFree(c.hist);
        cudaFree(c.ctl);
        c.hist = nullptr;
        c.ctl = nullptr;
        c.n_host = 0;
    }
    if (n_host <= 0) return 0;
    int prev;
    cudaGetDevice(&prev);
    cudaSetDevice(dev);
    bool ok = cudaMalloc((void**)&c.hist, (size_t)n_host * sizeof(long long)) == cudaSuccess &&
              cudaMalloc((void**)&c.ctl, 16 * sizeof(long long)) == cudaSuccess &&
              cudaMemset(c.hist, 0, (size_t)n_host * sizeof(long long)) == cudaSuccess &&
              cudaMemset(c.ctl, 0, 16 * sizeof(long long)) == cudaSuccess;
    cudaSetDevice(prev);
    if (!ok) {
        c.n_host = 0;
        return 1;
    }
    c.n_host = n_host;
    return 0;
}

/// Enqueues the counting kernel on `stream`. A device whose counters are gone (a close, a
/// resize that refused) counts nothing: the graphs that still name the op stay replayable.
void legion_kv_stats_launch(const kv_stats_args* a, void* stream) {
    const auto it = g_ctr.find(a->dev);
    if (it == g_ctr.end() || it->second.hist == nullptr) return;
    const long long total = (long long)a->rows * a->cap;
    if (total == 0) return;
    const int block = 256;
    const long long grid = (total + block - 1) / block;
    if (grid > 0x7fffffffLL) return;  // a selection that wide is a bug, not a measurement
    kv_stats_kernel<<<(int)grid, block, 0, (cudaStream_t)stream>>>(
        a->sel, a->sel_nb1, a->tab, a->tab_nb1, a->cap, a->rows, a->n_tok, a->hbm, a->n_host,
        a->off, it->second.hist, it->second.ctl);
}

/// Copies device `dev`'s counters out: `ctl` holds 16 values, `hist` the n_host the ensure
/// built. A CUDA error code, 0 for good; 3: the device has no counters.
int legion_kv_stats_read(int dev, unsigned long long* ctl, unsigned long long* hist) {
    const auto it = g_ctr.find(dev);
    if (it == g_ctr.end() || it->second.hist == nullptr) return 3;
    int prev;
    cudaGetDevice(&prev);
    cudaSetDevice(dev);
    bool ok = cudaMemcpy(ctl, it->second.ctl, 16 * sizeof(long long), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              cudaMemcpy(hist, it->second.hist, (size_t)it->second.n_host * sizeof(long long),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaSetDevice(prev);
    return ok ? 0 : 1;
}

/// The n_host device `dev`'s counters were built for, -1: none.
long long legion_kv_stats_n_host(int dev) {
    const auto it = g_ctr.find(dev);
    return it == g_ctr.end() || it->second.hist == nullptr ? -1 : it->second.n_host;
}

/// Zeroes device `dev`'s counters (a measurement run starts from here).
void legion_kv_stats_reset(int dev) {
    const auto it = g_ctr.find(dev);
    if (it == g_ctr.end() || it->second.hist == nullptr) return;
    int prev;
    cudaGetDevice(&prev);
    cudaSetDevice(dev);
    cudaMemset(it->second.ctl, 0, 16 * sizeof(long long));
    cudaMemset(it->second.hist, 0, (size_t)it->second.n_host * sizeof(long long));
    cudaSetDevice(prev);
}

}  // extern "C"
