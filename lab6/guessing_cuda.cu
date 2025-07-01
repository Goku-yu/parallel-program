
// guessing_cuda.cu - GPU加速版本（高性能优化版）

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>

#define MAX_LEN 64

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void generate_kernel(char *input, char *output, const char *prefix, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    char *in = input + i * MAX_LEN;
    char *out = output + i * MAX_LEN;

    int p = 0;
    if (prefix[0] != '\0') {
        while (prefix[p] != '\0' && p < MAX_LEN - 1) {
            out[p] = prefix[p];
            p++;
        }
    }

    int j = 0;
    while (in[j] != '\0' && p < MAX_LEN - 1) {
        out[p++] = in[j++];
    }
    out[p] = '\0';
}

extern "C" void gpu_generate(char *flat_input, int count, const std::string &prefix, std::vector<std::string> &result) {
    char *d_input = nullptr;
    char *d_output = nullptr;
    char *d_prefix = nullptr;
    size_t total_size = count * MAX_LEN;

    // 分配设备内存并拷贝输入
    CUDA_CHECK(cudaMalloc(&d_input, total_size));
    CUDA_CHECK(cudaMemcpy(d_input, flat_input, total_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_output, total_size));
    CUDA_CHECK(cudaMalloc(&d_prefix, MAX_LEN));
    CUDA_CHECK(cudaMemcpy(d_prefix, prefix.c_str(), prefix.size() + 1, cudaMemcpyHostToDevice));

    // 启动kernel
    int block = 256;
    int grid = (count + block - 1) / block;
    generate_kernel<<<grid, block>>>(d_input, d_output, d_prefix, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝结果回主机
    char *flat_output = new char[total_size];
    CUDA_CHECK(cudaMemcpy(flat_output, d_output, total_size, cudaMemcpyDeviceToHost));

    result.clear();
    for (int i = 0; i < count; ++i) {
        result.emplace_back(flat_output + i * MAX_LEN);
    }

    // 释放资源
    delete[] flat_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_prefix));
}
