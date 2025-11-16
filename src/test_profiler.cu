#include "utils/profiler.hpp"
#include "utils/device_query.hpp"
#include <iostream>
#include <iomanip>

// Simple vector addition kernel
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "GridAdvisor - Profiling Test\n";
    std::cout << "========================================\n";
    
    // Query device
    gridadvisor::DeviceQuery query;
    auto device = query.query(0);
    std::cout << "\nUsing GPU: " << device.name << "\n";
    
    // Problem size
    const int N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);
    
    std::cout << "Problem size: " << N << " elements (" 
              << (bytes / (1024.0 * 1024.0)) << " MB)\n";
    
    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Create profiler
    gridadvisor::KernelProfiler profiler(0);
    
    // Thread counts to test
    std::vector<int> thread_counts = {32, 64, 128, 256, 512, 1024};
    
    // Sweep configurations
    auto results = profiler.sweep_configurations(
        vector_add,
        N,
        thread_counts,
        5,   // warmup runs
        10,  // timing runs
        d_a, d_b, d_c, N
    );
    
    // Find best configuration
    auto best = std::min_element(results.begin(), results.end(),
        [](const gridadvisor::ProfilingResult& a, 
           const gridadvisor::ProfilingResult& b) {
            return a.mean_time_ms < b.mean_time_ms;
        });
    
    std::cout << "Best Configuration:\n";
    std::cout << "  Threads/Block: " << best->threads_per_block << "\n";
    std::cout << "  Num Blocks:    " << best->num_blocks << "\n";
    std::cout << "  Time:          " << best->mean_time_ms << " ms\n";
    std::cout << "  Std Dev:       " << best->std_time_ms << " ms\n";
    
    // Verify correctness with best config
    int blocks = best->num_blocks;
    int threads = best->threads_per_block;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "\nVerification: " << (correct ? "✓ PASSED" : "✗ FAILED") << "\n";
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    std::cout << "\n========================================\n";
    std::cout << "✓ Profiling test completed!\n";
    std::cout << "========================================\n\n";
    
    return 0;
}
