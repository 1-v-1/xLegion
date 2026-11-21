// src/cuda/cvec.cu - the speed projection's kernel and device tables, the CUDA side of
// src/model/qwen4/cvec.rs (`legion serve --speed-projection`, ported from Strata's
// `experimental-speed-projection`).
//
// After a steered layer's FFN write, every hyper-connection stream h of every token loses
// one direction: `h -= s · (h · v) · v` (`project` in Strata's `cvec.cu`: the unit direction
// `v` and its norm `s` in separate tables). One block per (stream, token) row, the row held
// in registers between the dot product and the update; the reduction is Strata's warp
// shuffle plus one shared-memory fold, so the two engines' sums differ only in order.
//
// The tables (`legion_cvec_store`) are kept on the host at load; `legion_cvec_ensure`
// uploads them to a device that lacks them (called at graph-build time, so a launch never
// allocates and a captured CUDA graph replays over pointers that never move).

#include <cuda_runtime.h>

#include <new>
#include <vector>

/// The projection's parameters (src/model/qwen4/cvec.rs, `CvecProj`). The streams are F32
/// `[e, hc, n]` contiguous, so stream `h` of token `t` is row `t · hc + h` of `e` values.
struct legion_cvec_proj {
    const float * res;
    float * dst;
    long long e;
    long long hc;
    long long n;
    int layer;
    int dev;
};

extern "C" {

/// Keeps host copies of the tables (one past the highest layer the vector carries), replacing
/// any earlier vector's tables on every device. 0 for good.
int legion_cvec_store(const float * dir, const float * s, int n_layers, long long n_embd);
/// Uploads the tables to device `dev` if it lacks them. A CUDA error code, 0 for good.
int legion_cvec_ensure(int dev);
/// Launches `dst = res - s · (res · v) · v` over the streams on `stream`. A CUDA error code.
int legion_cvec_project_launch(const legion_cvec_proj * p, void * stream);

}  // extern "C"

namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;  // e up to 4096, held in registers between the dot and the update
constexpr int WARP = 32;
constexpr int kDevices = 64;

struct Tables {
    float * dir = nullptr;  // n_layers · e: the unit directions
    float * s = nullptr;    // n_layers: their norms
};

std::vector<float> g_dir, g_s;  // the host copies a later device's ensure uploads from
int g_n_layers = 0;
long long g_n_embd = 0;
Tables g_dev[kDevices];

// one block per (stream, token) row: h . v over the row, then h -= s · (h . v) · v
__global__ void cvec_project_kernel(const float * __restrict__ res, float * __restrict__ dst,
                                    const float * __restrict__ dir, const float * __restrict__ s,
                                    const int e, const int hc, const long long rows,
                                    const int layer) {
    const float scale = s[layer];
    const float * __restrict__ v = dir + (long long) layer * e;
    for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const float * x = res + row * e;
        float * y = dst + row * e;
        float xv[MAXK];
        float dot = 0.0f;
#pragma unroll
        for (int k = 0; k < MAXK; ++k) {
            const int d = threadIdx.x + k * THREADS;
            if (d < e) {
                xv[k] = x[d];
                dot = fmaf(xv[k], v[d], dot);
            }
        }
        __shared__ float part[THREADS / WARP];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, o);
        }
        if ((threadIdx.x & (WARP - 1)) == 0) {
            part[threadIdx.x / WARP] = dot;
        }
        __syncthreads();
        if (threadIdx.x < WARP) {
            float p = threadIdx.x < THREADS / WARP ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) {
                p += __shfl_xor_sync(0xffffffffu, p, o);
            }
            if (threadIdx.x == 0) {
                part[0] = p;
            }
        }
        __syncthreads();
        const float k = scale * part[0];  // s (h . v)
#pragma unroll
        for (int j = 0; j < MAXK; ++j) {
            const int d = threadIdx.x + j * THREADS;
            if (d < e) {
                y[d] = fmaf(-k, v[d], xv[j]);
            }
        }
        // Strata's kernel owns one row per block and needs none of this; grid-striding rows
        // reuses `part`, so no warp may write it again before every thread read the sum.
        __syncthreads();
    }
}

}  // namespace

extern "C" int legion_cvec_store(const float * dir, const float * s, int n_layers,
                                 long long n_embd) {
    if (dir == nullptr || s == nullptr || n_layers < 1 || n_embd < 1 ||
        n_embd > (long long) THREADS * MAXK) {
        return (int) cudaErrorInvalidValue;
    }
    // A new vector replaces the old one's tables on every device; the engine builds no
    // graphs while `load` runs (it happens once, before serving).
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) {
            continue;
        }
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaFree(g_dev[d].dir);
        cudaFree(g_dev[d].s);
        g_dev[d] = Tables{};
    }
    cudaSetDevice(prev);
    g_dir.assign(dir, dir + (size_t) n_layers * (size_t) n_embd);
    g_s.assign(s, s + (size_t) n_layers);
    g_n_layers = n_layers;
    g_n_embd = n_embd;
    return (int) cudaSuccess;
}

extern "C" int legion_cvec_ensure(int dev) {
    if (dev < 0 || dev >= kDevices || g_n_layers == 0) {
        return (int) cudaErrorInvalidValue;
    }
    if (g_dev[dev].dir != nullptr) {
        return (int) cudaSuccess;
    }
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) {
        return (int) cudaErrorInvalidDevice;
    }
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaSetDevice(prev);
        return (int) cudaErrorInvalidDevice;
    }
    Tables t;
    int err = (int) cudaSuccess;
    if (cudaMalloc(&t.dir, g_dir.size() * sizeof(float)) != cudaSuccess) {
        err = (int) cudaGetLastError();
    } else if (cudaMalloc(&t.s, g_s.size() * sizeof(float)) != cudaSuccess) {
        cudaFree(t.dir);
        err = (int) cudaGetLastError();
    } else if (cudaMemcpy(t.dir, g_dir.data(), g_dir.size() * sizeof(float),
                          cudaMemcpyHostToDevice) != cudaSuccess ||
               cudaMemcpy(t.s, g_s.data(), g_s.size() * sizeof(float), cudaMemcpyHostToDevice) !=
                   cudaSuccess) {
        cudaFree(t.dir);
        cudaFree(t.s);
        err = (int) cudaGetLastError();
    } else {
        g_dev[dev] = t;
    }
    cudaSetDevice(prev);
    return err;
}

extern "C" int legion_cvec_project_launch(const legion_cvec_proj * p, void * stream) {
    if (p == nullptr || p->e < 1 || p->hc < 1 || p->n < 1 || p->dev < 0 ||
        p->dev >= kDevices || p->e != g_n_embd || (long long) p->layer >= (long long) g_n_layers) {
        return (int) cudaErrorInvalidValue;
    }
    const Tables & t = g_dev[p->dev];
    if (t.dir == nullptr) {
        // `project` uploads at graph-build time; a launch without it is a bug, not a miss.
        return (int) cudaErrorInvalidDevice;
    }
    const long long rows = p->hc * p->n;
    // Enough blocks to fill any current GPU; the kernel strides the rest (so a prompt step's
    // rows never meet the grid's x bound).
    const long long blocks = rows < 131072 ? rows : 131072;
    cvec_project_kernel<<<(unsigned) blocks, THREADS, 0, (cudaStream_t) stream>>>(
        p->res, p->dst, t.dir, t.s, (int) p->e, (int) p->hc, rows, p->layer);
    return (int) cudaGetLastError();
}
