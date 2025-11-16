#ifndef PROFILER_IMPL_HPP
#define PROFILER_IMPL_HPP

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace gridadvisor {

inline KernelProfiler::KernelProfiler(int dev_id) : device_id(dev_id) {
    cudaSetDevice(device_id);
}

inline KernelProfiler::~KernelProfiler() {
    cudaDeviceSynchronize();
}

inline ProfilingResult KernelProfiler::compute_statistics(
    const std::vector<float>& times,
    int threads,
    int blocks
) {
    ProfilingResult result;
    result.threads_per_block = threads;
    result.num_blocks = blocks;
    
    // Sort for median
    std::vector<float> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    
    // Mean
    float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    result.mean_time_ms = sum / times.size();
    
    // Std deviation
    float sq_sum = 0.0f;
    for (float t : times) {
        float diff = t - result.mean_time_ms;
        sq_sum += diff * diff;
    }
    result.std_time_ms = std::sqrt(sq_sum / times.size());
    
    // Min/max
    result.min_time_ms = sorted_times.front();
    result.max_time_ms = sorted_times.back();
    
    return result;
}

template<typename KernelFunc, typename... Args>
ProfilingResult KernelProfiler::profile(
    KernelFunc kernel,
    dim3 grid_size,
    dim3 block_size,
    size_t dynamic_smem,
    int num_warmup,
    int num_runs,
    Args... args
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    for (int i = 0; i < num_warmup; i++) {
        kernel<<<grid_size, block_size, dynamic_smem>>>(args...);
    }
    cudaDeviceSynchronize();
    
    // Timing runs
    std::vector<float> times;
    for (int i = 0; i < num_runs; i++) {
        cudaEventRecord(start);
        kernel<<<grid_size, block_size, dynamic_smem>>>(args...);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return compute_statistics(times, block_size.x, grid_size.x);
}

template<typename KernelFunc, typename... Args>
std::vector<ProfilingResult> KernelProfiler::sweep_configurations(
    KernelFunc kernel,
    int problem_size,
    const std::vector<int>& thread_counts,
    int num_warmup,
    int num_runs,
    Args... args
) {
    std::vector<ProfilingResult> results;
    
    std::cout << "\nSweeping Configurations:\n";
    std::cout << "Problem size: " << problem_size << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Threads/Block | Num Blocks | Mean Time (ms) | Speedup\n";
    std::cout << std::string(60, '-') << "\n";
    
    float baseline_time = 0.0f;
    
    for (int threads : thread_counts) {
        int blocks = (problem_size + threads - 1) / threads;
        
        dim3 grid(blocks, 1, 1);
        dim3 block(threads, 1, 1);
        
        ProfilingResult result = profile(
            kernel, grid, block, 0,
            num_warmup, num_runs,
            args...
        );
        
        if (baseline_time == 0.0f) {
            baseline_time = result.mean_time_ms;
        }
        
        float speedup = baseline_time / result.mean_time_ms;
        
        printf("%13d | %10d | %14.4f | %7.2fx\n",
               threads, blocks, result.mean_time_ms, speedup);
        
        results.push_back(result);
    }
    
    std::cout << std::string(60, '-') << "\n\n";
    
    return results;
}

} // namespace gridadvisor

#endif
