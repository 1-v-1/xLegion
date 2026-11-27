// src/cuda/kv_stats.cu - the KV slot cache's Phase 0 measurement (the CUDA side of
// src/model/qwen4/kv_stats.rs, docs/kv-streaming-porting-notes.md): count where the QSA
// selections land, per block, on a pool that has a host tier.
//
// The counting is what `after_attn` does after a step's attention launch (next to the
// selection dump of sel_dump.rs): one extra kernel on the step's stream, behind nothing
// the model reads. No graph node, no tensor, no allocation: the graph is the same graph
// with the measurement on or off, and the kernel reads only the selection rows and block
// table and writes only the counters, so the model's bits are untouched.
//
// The counters live in device memory at fixed addresses, one set per device (the registry
// pattern of cvec.cu: `legion_kv_stats_ensure` builds them at pool-build time, and a
// launch only reads the map), so the captured graphs bake no address the load could move.
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

namespace {

struct Ctr {
    unsigned long long* hist = nullptr;  // (n_host,) fetches per host block
    unsigned long long* ctl = nullptr;   // 16 u64
    long long n_host = 0;
    long long hbm = 0;  // the pool's first host block (the registered one, not the baked)
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
    if (h >= n_host) return;  // the pool moved since: totals only
    // The positions are ascending, so an earlier cell of the same row says this block was
    // already fetched for this token: the histogram counts each fetch once, as a slot pool
    // would move it once.
    if (slot > 0 && r[slot] / 64 == pos / 64) return;
    atomicAdd(&hist[h], 1ULL);
    atomicAdd(&ctl[off + 2], 1ULL);
}

}  // namespace

extern "C" {

/// Builds device `dev`'s counters for a pool whose host range is `n_host` blocks wide
/// starting at block `hbm`, zeroed. A resize moves the host range's block ids, so a
/// changed `n_host` starts the counters over; a pool rebuilt at the same size keeps them.
/// A CUDA error code, 0 for good.
int legion_kv_stats_ensure(int dev, long long n_host, long long hbm) {
    Ctr& c = g_ctr[dev];
    if (c.hist != nullptr && c.n_host == n_host) {
        c.hbm = hbm;  // the same host size at a new boundary: the kernel reads the latest
        return 0;
    }
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
    c.hbm = hbm;
    return 0;
}

/// Enqueues the counting kernel on `stream` for the step's selection rows
/// (`rows` rows of `sel_nb1` I32s, cap = sel_nb1 - 1) and block tables (`tab_nb1`
/// = the table width). A device with no registered pool counts nothing: the hook needs
/// no branch at the call site beyond the switch.
void legion_kv_stats_after(const int* sel, long long sel_nb1, const int* tab, long long tab_nb1,
                           int rows, int n_tok, int off, void* stream) {
    int dev;
    if (cudaGetDevice(&dev) != cudaSuccess) return;
    const auto it = g_ctr.find(dev);
    if (it == g_ctr.end() || it->second.hist == nullptr) return;
    const int cap = (int)sel_nb1 - 1;
    const long long total = (long long)rows * cap;
    if (total <= 0) return;
    const int block = 256;
    const long long grid = (total + block - 1) / block;
    if (grid > 0x7fffffffLL) return;  // a selection that wide is a bug, not a measurement
    kv_stats_kernel<<<(int)grid, block, 0, (cudaStream_t)stream>>>(
        sel, sel_nb1, tab, tab_nb1, cap, rows, n_tok, it->second.hbm, it->second.n_host, off,
        it->second.hist, it->second.ctl);
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
