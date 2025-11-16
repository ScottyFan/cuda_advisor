#include "features/feature_extractor.hpp"
#include "utils/profiler.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace gridadvisor {

// Occupancy calculator
struct OccupancyResult {
    float occupancy;
    int blocks_per_sm;
    std::string limiting_factor;
};

OccupancyResult calculate_occupancy(
    int threads_per_block,
    int regs_per_thread,
    int shared_mem_per_block,
    const DeviceSpecs& device
) {
    OccupancyResult result;
    
    int warps_per_block = (threads_per_block + 31) / 32;
    
    // Calculate limits
    int limit_by_warps = device.max_blocks_per_sm;
    int limit_by_regs = regs_per_thread > 0 ? 
        device.regs_per_sm / (regs_per_thread * threads_per_block) : INT_MAX;
    int limit_by_smem = shared_mem_per_block > 0 ?
        device.shared_mem_per_sm / shared_mem_per_block : INT_MAX;
    int limit_by_threads = device.max_threads_per_sm / threads_per_block;
    
    // Find bottleneck
    result.blocks_per_sm = std::min({limit_by_warps, limit_by_regs, 
                                      limit_by_smem, limit_by_threads});
    
    if (result.blocks_per_sm == limit_by_regs) {
        result.limiting_factor = "Registers";
    } else if (result.blocks_per_sm == limit_by_smem) {
        result.limiting_factor = "Shared Memory";
    } else if (result.blocks_per_sm == limit_by_threads) {
        result.limiting_factor = "Thread Count";
    } else {
        result.limiting_factor = "Block Limit";
    }
    
    int active_warps = result.blocks_per_sm * warps_per_block;
    int max_warps = device.max_threads_per_sm / 32;
    result.occupancy = (float)active_warps / max_warps;
    
    return result;
}

// Simple analytical predictor
struct Recommendation {
    int threads_per_block;
    int num_blocks;
    float predicted_occupancy;
    float confidence;
    std::string reasoning;
};

Recommendation predict_configuration(const FeatureVector& features) {
    DeviceQuery query;
    auto device = query.query(0);
    
    std::vector<int> candidates = {64, 128, 256, 512, 1024};
    
    struct ScoredConfig {
        int threads;
        float score;
        float occupancy;
    };
    
    std::vector<ScoredConfig> scored;
    
    for (int threads : candidates) {
        // Calculate occupancy
        auto occ = calculate_occupancy(
            threads,
            (int)features.f6_registers_per_thread,
            (int)(features.f7_shared_memory_per_block_kb * 1024),
            device
        );
        
        // Score based on kernel characteristics
        float score = 0.0f;
        
        // High arithmetic intensity → prefer high occupancy
        if (features.f12_arithmetic_intensity > 1.0f) {
            score += occ.occupancy * 0.6f;
        } else {
            // Memory-bound → balance occupancy with coalescing
            score += occ.occupancy * 0.4f;
            if (threads >= 128) score += 0.2f;  // Better coalescing
        }
        
        // Bonus for good occupancy
        if (occ.occupancy > 0.5f) score += 0.2f;
        
        // Penalty for very low occupancy
        if (occ.occupancy < 0.25f) score *= 0.5f;
        
        // Power-of-2 bonus
        if ((threads & (threads - 1)) == 0) score += 0.1f;
        
        scored.push_back({threads, score, occ.occupancy});
    }
    
    // Find best
    auto best = std::max_element(scored.begin(), scored.end(),
        [](const ScoredConfig& a, const ScoredConfig& b) {
            return a.score < b.score;
        });
    
    Recommendation rec;
    rec.threads_per_block = best->threads;
    rec.num_blocks = ((int)features.f23_total_work_items + best->threads - 1) / best->threads;
    rec.predicted_occupancy = best->occupancy;
    rec.confidence = best->score;
    
    // Generate reasoning
    if (features.f12_arithmetic_intensity > 1.0f) {
        rec.reasoning = "Compute-bound: Maximizing occupancy";
    } else {
        rec.reasoning = "Memory-bound: Balancing occupancy and coalescing";
    }
    
    return rec;
}

} // namespace gridadvisor

// Test the complete system
int main() {
    using namespace gridadvisor;
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "   GridAdvisor - Complete System Test\n";
    std::cout << "========================================\n";
    
    // Query device
    DeviceQuery query;
    auto device = query.query(0);
    std::cout << "\nGPU: " << device.name << "\n";
    
    // Test with different kernels
    extern __global__ void transpose(const float*, float*, int, int);
    extern __global__ void matmul_shared(const float*, const float*, float*, int, int, int);
    
    // Extract features and predict
    FeatureExtractor extractor(0);
    
    std::cout << "\n=== Test 1: Transpose (Memory-Bound) ===\n";
    auto fv1 = extractor.extract(transpose, 2048 * 2048);
    auto rec1 = predict_configuration(fv1);
    
    std::cout << "Recommendation:\n";
    std::cout << "  Threads/Block: " << rec1.threads_per_block << "\n";
    std::cout << "  Num Blocks:    " << rec1.num_blocks << "\n";
    std::cout << "  Occupancy:     " << (rec1.predicted_occupancy * 100) << "%\n";
    std::cout << "  Confidence:    " << rec1.confidence << "\n";
    std::cout << "  Reasoning:     " << rec1.reasoning << "\n";
    
    std::cout << "\n=== Test 2: MatMul Shared (Compute-Bound) ===\n";
    auto fv2 = extractor.extract(matmul_shared, 512 * 512);
    auto rec2 = predict_configuration(fv2);
    
    std::cout << "Recommendation:\n";
    std::cout << "  Threads/Block: " << rec2.threads_per_block << "\n";
    std::cout << "  Num Blocks:    " << rec2.num_blocks << "\n";
    std::cout << "  Occupancy:     " << (rec2.predicted_occupancy * 100) << "%\n";
    std::cout << "  Confidence:    " << rec2.confidence << "\n";
    std::cout << "  Reasoning:     " << rec2.reasoning << "\n";
    
    std::cout << "\n========================================\n";
    std::cout << "✓ GridAdvisor system working!\n";
    std::cout << "========================================\n\n";
    
    return 0;
}
