#ifndef FEATURE_EXTRACTOR_IMPL_HPP
#define FEATURE_EXTRACTOR_IMPL_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

namespace gridadvisor {

inline FeatureExtractor::FeatureExtractor(int dev_id) : device_id(dev_id) {
    DeviceQuery query;
    device_specs = query.query(device_id);
}

template<typename KernelFunc>
KernelCharacteristics FeatureExtractor::get_kernel_characteristics(KernelFunc kernel) {
    KernelCharacteristics kc;
    
    // Get function attributes using CUDA Driver API
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    
    kc.registers_per_thread = attr.numRegs;
    kc.shared_memory_static = attr.sharedSizeBytes;
    kc.const_memory = attr.constSizeBytes;
    kc.local_memory = attr.localSizeBytes;
    kc.max_threads_per_block = attr.maxThreadsPerBlock;
    
    // Get kernel name (simplified)
    kc.name = "kernel";
    
    // Detect characteristics based on resource usage
    kc.uses_shared_memory = (attr.sharedSizeBytes > 0);
    kc.has_atomics = false;  // Can't detect easily
    
    // Classify kernel type based on heuristics
    if (kc.shared_memory_static > 16384) {
        // High shared memory usage suggests compute-bound (tiling, etc.)
        kc.type = KernelCharacteristics::Type::COMPUTE_BOUND;
        kc.estimated_ai = 2.0f;  // High arithmetic intensity
    } else if (kc.registers_per_thread < 32) {
        // Low register usage suggests memory-bound
        kc.type = KernelCharacteristics::Type::MEMORY_BOUND;
        kc.estimated_ai = 0.5f;  // Low arithmetic intensity
    } else {
        kc.type = KernelCharacteristics::Type::BALANCED;
        kc.estimated_ai = 1.0f;
    }
    
    return kc;
}

template<typename KernelFunc>
FeatureVector FeatureExtractor::extract(KernelFunc kernel, int problem_size) {
    FeatureVector fv = {};
    
    // Get kernel characteristics
    auto kc = get_kernel_characteristics(kernel);
    
    // Hardware features (from device specs)
    fv.f1_sm_count = device_specs.sm_count;
    fv.f2_max_threads_per_sm = device_specs.max_threads_per_sm;
    fv.f3_memory_bandwidth_gb_s = device_specs.memory_bandwidth_gb_s;
    fv.f4_peak_gflops = device_specs.peak_gflops_fp32;
    fv.f5_shared_memory_per_sm_kb = device_specs.shared_mem_per_sm / 1024.0f;
    
    // Kernel resource features
    fv.f6_registers_per_thread = kc.registers_per_thread;
    fv.f7_shared_memory_per_block_kb = kc.shared_memory_static / 1024.0f;
    fv.f8_local_memory_per_thread_b = kc.local_memory;
    fv.f9_const_memory_kb = kc.const_memory / 1024.0f;
    fv.f10_instruction_count = 100.0f;  // Placeholder estimate
    fv.f11_binary_size_kb = 10.0f;      // Placeholder estimate
    
    // Compute characteristics (estimated)
    estimate_compute_features(fv, kc);
    
    // Memory access features (estimated)
    estimate_memory_features(fv, kc);
    
    // Control flow features (heuristics)
    fv.f21_branch_intensity = 0.1f;  // Default low
    fv.f22_sync_intensity = kc.uses_shared_memory ? 0.2f : 0.0f;
    
    // Problem size features
    fv.f23_total_work_items = problem_size;
    fv.f24_work_per_sm = static_cast<float>(problem_size) / device_specs.sm_count;
    
    return fv;
}

inline void FeatureExtractor::estimate_compute_features(
    FeatureVector& fv, 
    const KernelCharacteristics& kc
) {
    // Estimate based on kernel type
    fv.f12_arithmetic_intensity = kc.estimated_ai;
    
    switch (kc.type) {
        case KernelCharacteristics::Type::COMPUTE_BOUND:
            fv.f13_flop_count = 1000.0f;  // High
            fv.f14_memory_ops = 100.0f;
            fv.f15_fp_ratio = 0.8f;
            fv.f16_transcendental_ratio = 0.1f;
            break;
            
        case KernelCharacteristics::Type::MEMORY_BOUND:
            fv.f13_flop_count = 100.0f;   // Low
            fv.f14_memory_ops = 1000.0f;
            fv.f15_fp_ratio = 0.3f;
            fv.f16_transcendental_ratio = 0.0f;
            break;
            
        default:
            fv.f13_flop_count = 500.0f;
            fv.f14_memory_ops = 500.0f;
            fv.f15_fp_ratio = 0.5f;
            fv.f16_transcendental_ratio = 0.05f;
    }
}

inline void FeatureExtractor::estimate_memory_features(
    FeatureVector& fv,
    const KernelCharacteristics& kc
) {
    // Estimate access pattern
    if (kc.uses_shared_memory) {
        fv.f17_access_pattern_score = 0.8f;  // Likely coalesced if using shared mem
        fv.f18_shared_mem_reuse = 2.0f;      // Good reuse
    } else {
        fv.f17_access_pattern_score = 0.5f;  // Unknown
        fv.f18_shared_mem_reuse = 0.0f;
    }
    
    fv.f19_global_load_ratio = 0.5f;  // Assume balanced loads/stores
    fv.f20_memory_divergence = 0.1f;  // Low divergence default
}

inline void FeatureVector::print() const {
    std::cout << "\n=== Feature Vector (24 dimensions) ===\n\n";
    
    std::cout << "Hardware Features:\n";
    std::cout << "  f1_sm_count:                 " << f1_sm_count << "\n";
    std::cout << "  f2_max_threads_per_sm:       " << f2_max_threads_per_sm << "\n";
    std::cout << "  f3_memory_bandwidth_gb_s:    " << f3_memory_bandwidth_gb_s << "\n";
    std::cout << "  f4_peak_gflops:              " << f4_peak_gflops << "\n";
    std::cout << "  f5_shared_memory_per_sm_kb:  " << f5_shared_memory_per_sm_kb << "\n";
    
    std::cout << "\nKernel Resource Features:\n";
    std::cout << "  f6_registers_per_thread:     " << f6_registers_per_thread << "\n";
    std::cout << "  f7_shared_memory_per_block_kb: " << f7_shared_memory_per_block_kb << "\n";
    std::cout << "  f8_local_memory_per_thread_b: " << f8_local_memory_per_thread_b << "\n";
    std::cout << "  f9_const_memory_kb:          " << f9_const_memory_kb << "\n";
    std::cout << "  f10_instruction_count:       " << f10_instruction_count << "\n";
    std::cout << "  f11_binary_size_kb:          " << f11_binary_size_kb << "\n";
    
    std::cout << "\nCompute Characteristics:\n";
    std::cout << "  f12_arithmetic_intensity:    " << f12_arithmetic_intensity << "\n";
    std::cout << "  f13_flop_count:              " << f13_flop_count << "\n";
    std::cout << "  f14_memory_ops:              " << f14_memory_ops << "\n";
    std::cout << "  f15_fp_ratio:                " << f15_fp_ratio << "\n";
    std::cout << "  f16_transcendental_ratio:    " << f16_transcendental_ratio << "\n";
    
    std::cout << "\nMemory Access Features:\n";
    std::cout << "  f17_access_pattern_score:    " << f17_access_pattern_score << "\n";
    std::cout << "  f18_shared_mem_reuse:        " << f18_shared_mem_reuse << "\n";
    std::cout << "  f19_global_load_ratio:       " << f19_global_load_ratio << "\n";
    std::cout << "  f20_memory_divergence:       " << f20_memory_divergence << "\n";
    
    std::cout << "\nControl Flow Features:\n";
    std::cout << "  f21_branch_intensity:        " << f21_branch_intensity << "\n";
    std::cout << "  f22_sync_intensity:          " << f22_sync_intensity << "\n";
    
    std::cout << "\nProblem Size Features:\n";
    std::cout << "  f23_total_work_items:        " << f23_total_work_items << "\n";
    std::cout << "  f24_work_per_sm:             " << f24_work_per_sm << "\n";
    std::cout << "\n";
}

inline std::vector<float> FeatureVector::to_array() const {
    return {
        f1_sm_count, f2_max_threads_per_sm, f3_memory_bandwidth_gb_s,
        f4_peak_gflops, f5_shared_memory_per_sm_kb,
        f6_registers_per_thread, f7_shared_memory_per_block_kb,
        f8_local_memory_per_thread_b, f9_const_memory_kb,
        f10_instruction_count, f11_binary_size_kb,
        f12_arithmetic_intensity, f13_flop_count, f14_memory_ops,
        f15_fp_ratio, f16_transcendental_ratio,
        f17_access_pattern_score, f18_shared_mem_reuse,
        f19_global_load_ratio, f20_memory_divergence,
        f21_branch_intensity, f22_sync_intensity,
        f23_total_work_items, f24_work_per_sm
    };
}

} // namespace gridadvisor

#endif
