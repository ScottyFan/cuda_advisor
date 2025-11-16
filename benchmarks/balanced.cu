// Balanced compute/memory kernels
#include <cuda_runtime.h>

// 1. 2D Stencil (3x3 convolution)
__global__ void stencil_2d(const float* input, float* output, 
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        float sum = 0.0f;
        // 3x3 stencil
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += input[(y + dy) * width + (x + dx)];
            }
        }
        output[y * width + x] = sum / 9.0f;
    }
}

// 2. Histogram (atomic operations)
__global__ void histogram(const unsigned char* input, unsigned int* hist,
                          int n, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int bin = (input[idx] * num_bins) / 256;
        atomicAdd(&hist[bin], 1);
    }
}

// 3. Prefix sum (scan) - simplified
__global__ void prefix_sum(const float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    temp[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
    }
    __syncthreads();
    
    if (idx < n) {
        output[idx] = temp[tid];
    }
}