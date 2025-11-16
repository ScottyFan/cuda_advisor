#include "utils/profiler.hpp"
#include "utils/device_query.hpp"
#include <iostream>
#include <vector>

// Declare kernels from benchmark files
extern __global__ void transpose(const float*, float*, int, int);
extern __global__ void reduction_sum(const float*, float*, int);
extern __global__ void matmul_naive(const float*, const float*, float*, int, int, int);
extern __global__ void matmul_shared(const float*, const float*, float*, int, int, int);
extern __global__ void stencil_2d(const float*, float*, int, int);

void test_transpose() {
    std::cout << "\n=== Testing Transpose ===\n";
    
    const int N = 2048;
    const size_t bytes = N * N * sizeof(float);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    gridadvisor::KernelProfiler profiler(0);
    std::vector<int> thread_counts = {64, 128, 256, 512};
    
    auto results = profiler.sweep_configurations(
        transpose, N * N, thread_counts, 3, 5,
        d_in, d_out, N, N
    );
    
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_reduction() {
    std::cout << "\n=== Testing Reduction ===\n";
    
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, 65536 * sizeof(float));  // Max blocks
    
    gridadvisor::KernelProfiler profiler(0);
    std::vector<int> thread_counts = {128, 256, 512, 1024};
    
    // Note: reduction uses dynamic shared memory
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "Problem size: " << N << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Threads/Block | Shared Mem | Mean Time (ms)\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (int threads : thread_counts) {
        int blocks = (N + threads - 1) / threads;
        blocks = std::min(blocks, 65536);  // Limit blocks
        
        size_t smem_size = threads * sizeof(float);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            reduction_sum<<<blocks, threads, smem_size>>>(d_in, d_out, N);
        }
        cudaDeviceSynchronize();
        
        // Time
        std::vector<float> times;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(start);
            reduction_sum<<<blocks, threads, smem_size>>>(d_in, d_out, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times.push_back(ms);
        }
        
        float mean = 0.0f;
        for (float t : times) mean += t;
        mean /= times.size();
        
        printf("%13d | %9.1f KB | %14.4f\n", 
               threads, smem_size/1024.0f, mean);
    }
    
    std::cout << std::string(60, '-') << "\n";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_matmul() {
    std::cout << "\n=== Testing Matrix Multiply ===\n";
    
    const int M = 512, N = 512, K = 512;
    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    // Test naive version
    std::cout << "\nNaive version:\n";
    gridadvisor::KernelProfiler profiler(0);
    std::vector<int> thread_counts = {64, 128, 256};
    
    auto results = profiler.sweep_configurations(
        matmul_naive, M * N, thread_counts, 2, 3,
        d_A, d_B, d_C, M, N, K
    );
    
    // Test shared memory version (fixed block size)
    std::cout << "\nShared memory version (16x16 blocks):\n";
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 2; i++) {
        matmul_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    // Time
    std::vector<float> times;
    for (int i = 0; i < 3; i++) {
        cudaEventRecord(start);
        matmul_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    
    float mean = 0.0f;
    for (float t : times) mean += t;
    mean /= times.size();
    
    printf("Block: 16x16, Grid: %dx%d, Time: %.4f ms\n", 
           grid.x, grid.y, mean);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_stencil() {
    std::cout << "\n=== Testing Stencil 2D ===\n";
    
    const int width = 2048, height = 2048;
    const size_t bytes = width * height * sizeof(float);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    gridadvisor::KernelProfiler profiler(0);
    std::vector<int> thread_counts = {64, 128, 256, 512};
    
    auto results = profiler.sweep_configurations(
        stencil_2d, width * height, thread_counts, 3, 5,
        d_in, d_out, width, height
    );
    
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "GridAdvisor - Benchmark Suite Test\n";
    std::cout << "========================================\n";
    
    // Query device
    gridadvisor::DeviceQuery query;
    auto device = query.query(0);
    std::cout << "\nGPU: " << device.name << "\n";
    std::cout << "SM Count: " << device.sm_count << "\n";
    std::cout << "Memory Bandwidth: " << device.memory_bandwidth_gb_s << " GB/s\n";
    
    // Run all benchmarks
    test_transpose();
    test_reduction();
    test_matmul();
    test_stencil();
    
    std::cout << "\n========================================\n";
    std::cout << "✓ All benchmarks completed!\n";
    std::cout << "========================================\n\n";
    
    return 0;
}
