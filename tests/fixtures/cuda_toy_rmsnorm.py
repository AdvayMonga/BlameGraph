"""A CUDA C++ program for toy_rmsnorm, built with load_inline: what the GPU test judges through the jail."""
from torch.utils.cpp_extension import load_inline

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void rmsnorm_k(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int dim,
                          float eps) {
    const float* xr = x + (size_t)blockIdx.x * dim;
    float* yr = y + (size_t)blockIdx.x * dim;
    __shared__ float part[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (threadIdx.x == 0) part[0] = rsqrtf(s / dim + eps);
    }
    __syncthreads();
    float r = part[0];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
    auto xc = x.contiguous();
    auto wc = w.contiguous();
    auto y = torch::empty_like(xc);
    int rows = xc.size(0), dim = xc.size(1);
    if (rows == 0) return y;
    rmsnorm_k<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(xc.data_ptr<float>(), wc.data_ptr<float>(),
                                                                 y.data_ptr<float>(), dim, (float)eps);
    return y;
}
"""

_ext = load_inline("toy_rmsnorm_cuda", cpp_sources="torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps);",
                   cuda_sources=CUDA, functions=["rmsnorm"], extra_cuda_cflags=["-O3"])


def kernel(x, w):
    return _ext.rmsnorm(x, w, 1e-6)
