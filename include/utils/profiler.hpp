#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <cuda_runtime.h>
#include <vector>

namespace gridadvisor {

struct ProfilingResult {
    float mean_time_ms;
    float std_time_ms;
    float min_time_ms;
    float max_time_ms;
    int threads_per_block;
    int num_blocks;
};

class KernelProfiler {
public:
    KernelProfiler(int device_id = 0);
    ~KernelProfiler();
    
    // Profile a kernel with specific configuration
    template<typename KernelFunc, typename... Args>
    ProfilingResult profile(
        KernelFunc kernel,
        dim3 grid_size,
        dim3 block_size,
        size_t dynamic_smem,
        int num_warmup,
        int num_runs,
        Args... args
    );
    
    // Sweep multiple thread counts for a kernel
    template<typename KernelFunc, typename... Args>
    std::vector<ProfilingResult> sweep_configurations(
        KernelFunc kernel,
        int problem_size,
        const std::vector<int>& thread_counts,
        int num_warmup,
        int num_runs,
        Args... args
    );
    
private:
    int device_id;
    ProfilingResult compute_statistics(const std::vector<float>& times, 
                                       int threads, int blocks);
};

} // namespace gridadvisor

#include "profiler_impl.hpp"  // Template implementations

#endif
