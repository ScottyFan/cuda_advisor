// Compute-bound benchmark kernels
#include <cuda_runtime.h>

// 1. Matrix Multiply - Naive (high arithmetic intensity)
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. Matrix Multiply - Shared Memory (optimized)
#define TILE_SIZE 16

__global__ void matmul_shared(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? 
            A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? 
            B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 3. Mandelbrot set computation (FP-heavy)
__global__ void mandelbrot(float* output, int width, int height, 
                           int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float c_re = (x - width/2.0f) * 4.0f / width;
    float c_im = (y - height/2.0f) * 4.0f / height;
    
    float z_re = 0.0f, z_im = 0.0f;
    int iter = 0;
    
    while (z_re * z_re + z_im * z_im < 4.0f && iter < max_iter) {
        float temp = z_re * z_re - z_im * z_im + c_re;
        z_im = 2.0f * z_re * z_im + c_im;
        z_re = temp;
        iter++;
    }
    
    output[y * width + x] = (float)iter / max_iter;
}