#include "utils/device_query.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

namespace gridadvisor {

DeviceQuery::DeviceQuery() {
    // Initialize CUDA
    int device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count), "Get device count");
    
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices found!");
    }
}

DeviceQuery::~DeviceQuery() {
    cudaDeviceReset();
}

void DeviceQuery::check_cuda_error(int error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " 
                  << cudaGetErrorString((cudaError_t)error) << std::endl;
        throw std::runtime_error(msg);
    }
}

int DeviceQuery::get_device_count() {
    int count;
    check_cuda_error(cudaGetDeviceCount(&count), "Get device count");
    return count;
}

DeviceSpecs DeviceQuery::query(int device_id) {
    DeviceSpecs specs;
    cudaDeviceProp prop;
    
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), 
                     "Get device properties");
    
    // Basic info
    specs.name = std::string(prop.name);
    specs.device_id = device_id;
    
    // Compute capability
    specs.compute_major = prop.major;
    specs.compute_minor = prop.minor;
    
    // Threading
    specs.sm_count = prop.multiProcessorCount;
    specs.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    specs.max_threads_per_block = prop.maxThreadsPerBlock;
    specs.warp_size = prop.warpSize;
    
    // Max blocks per SM depends on architecture
    // This is a simplified estimate
    if (specs.compute_major >= 8) {
        specs.max_blocks_per_sm = 32;  // Ampere, Hopper
    } else if (specs.compute_major >= 7) {
        specs.max_blocks_per_sm = 32;  // Volta, Turing
    } else {
        specs.max_blocks_per_sm = 16;  // Older architectures
    }
    
    // Memory
    specs.global_memory_bytes = prop.totalGlobalMem;
    specs.shared_mem_per_block = prop.sharedMemPerBlock;
    specs.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
    specs.const_memory_bytes = prop.totalConstMem;
    specs.l2_cache_size = prop.l2CacheSize;
    
    // Registers
    specs.regs_per_block = prop.regsPerBlock;
    specs.regs_per_sm = prop.regsPerMultiprocessor;
    
    // Clocks and bandwidth
    specs.memory_clock_mhz = prop.memoryClockRate / 1000.0f;
    specs.memory_bus_width = prop.memoryBusWidth;
    specs.clock_rate_mhz = prop.clockRate / 1000.0f;
    
    // Calculate memory bandwidth (GB/s)
    // Bandwidth = (Memory Clock * 2) * (Bus Width / 8) / 1e9
    // The factor of 2 is for DDR (Double Data Rate)
    specs.memory_bandwidth_gb_s = 
        2.0f * (prop.memoryClockRate / 1.0e6) * (prop.memoryBusWidth / 8.0f);
    
    // Estimate peak GFlops
    // This is architecture-dependent and simplified
    // FP32 cores per SM varies by architecture
    int fp32_cores_per_sm;
    if (specs.compute_major == 8 && specs.compute_minor == 0) {
        fp32_cores_per_sm = 64;  // A100
    } else if (specs.compute_major == 8 && specs.compute_minor == 6) {
        fp32_cores_per_sm = 128; // A40, A10
    } else if (specs.compute_major == 7 && specs.compute_minor == 5) {
        fp32_cores_per_sm = 64;  // T4, Turing
    } else if (specs.compute_major == 7 && specs.compute_minor == 0) {
        fp32_cores_per_sm = 64;  // V100
    } else {
        fp32_cores_per_sm = 128; // Default estimate
    }
    
    // Peak GFlops = SM_count * cores_per_SM * clock_GHz * 2 (FMA)
    specs.peak_gflops_fp32 = 
        specs.sm_count * fp32_cores_per_sm * (specs.clock_rate_mhz / 1000.0f) * 2.0f;
    
    return specs;
}

std::vector<DeviceSpecs> DeviceQuery::query_all() {
    std::vector<DeviceSpecs> all_specs;
    int count = get_device_count();
    
    for (int i = 0; i < count; i++) {
        all_specs.push_back(query(i));
    }
    
    return all_specs;
}

void DeviceSpecs::print() const {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "GPU Device " << device_id << ": " << name << "\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\nCompute Capability:\n";
    std::cout << "  Version:              " << compute_major << "." << compute_minor << "\n";
    
    std::cout << "\nThreading:\n";
    std::cout << "  SM Count:             " << sm_count << "\n";
    std::cout << "  Max Threads/SM:       " << max_threads_per_sm << "\n";
    std::cout << "  Max Threads/Block:    " << max_threads_per_block << "\n";
    std::cout << "  Max Blocks/SM:        " << max_blocks_per_sm << "\n";
    std::cout << "  Warp Size:            " << warp_size << "\n";
    
    std::cout << "\nMemory:\n";
    std::cout << "  Global Memory:        " 
              << (global_memory_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Shared Mem/Block:     " 
              << (shared_mem_per_block / 1024.0) << " KB\n";
    std::cout << "  Shared Mem/SM:        " 
              << (shared_mem_per_sm / 1024.0) << " KB\n";
    std::cout << "  Constant Memory:      " 
              << (const_memory_bytes / 1024.0) << " KB\n";
    std::cout << "  L2 Cache:             " 
              << (l2_cache_size / (1024.0 * 1024.0)) << " MB\n";
    
    std::cout << "\nRegisters:\n";
    std::cout << "  Registers/Block:      " << regs_per_block << "\n";
    std::cout << "  Registers/SM:         " << regs_per_sm << "\n";
    
    std::cout << "\nPerformance:\n";
    std::cout << "  Clock Rate:           " << clock_rate_mhz << " MHz\n";
    std::cout << "  Memory Clock:         " << memory_clock_mhz << " MHz\n";
    std::cout << "  Memory Bus Width:     " << memory_bus_width << " bits\n";
    std::cout << "  Memory Bandwidth:     " << memory_bandwidth_gb_s << " GB/s\n";
    std::cout << "  Peak FP32 GFlops:     " << peak_gflops_fp32 << "\n";
    
    std::cout << "========================================\n\n";
}

void DeviceSpecs::to_json(const std::string& filename) const {
    std::ofstream file(filename);
    
    file << "{\n";
    file << "  \"device_id\": " << device_id << ",\n";
    file << "  \"name\": \"" << name << "\",\n";
    file << "  \"compute_capability\": \"" << compute_major << "." << compute_minor << "\",\n";
    file << "  \"sm_count\": " << sm_count << ",\n";
    file << "  \"max_threads_per_sm\": " << max_threads_per_sm << ",\n";
    file << "  \"max_threads_per_block\": " << max_threads_per_block << ",\n";
    file << "  \"max_blocks_per_sm\": " << max_blocks_per_sm << ",\n";
    file << "  \"warp_size\": " << warp_size << ",\n";
    file << "  \"global_memory_gb\": " << (global_memory_bytes / (1024.0 * 1024.0 * 1024.0)) << ",\n";
    file << "  \"shared_mem_per_sm_kb\": " << (shared_mem_per_sm / 1024.0) << ",\n";
    file << "  \"regs_per_sm\": " << regs_per_sm << ",\n";
    file << "  \"memory_bandwidth_gb_s\": " << memory_bandwidth_gb_s << ",\n";
    file << "  \"peak_gflops_fp32\": " << peak_gflops_fp32 << "\n";
    file << "}\n";
    
    file.close();
}

} // namespace gridadvisor
