/*
 * CUDA Financial Risk Engine (v2.0 - Fixed)
 * Features:
 * 1. Monte Carlo Option Pricing (GPU)
 * 2. Greek Calculation (Delta for Hedging)
 * 3. CPU vs GPU Benchmarking
 * 4. CSV Data Export (Restored)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random> // For CPU benchmark

// CUDA & Libraries
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/common.h"

// Forward Declaration
__global__ void initRNG(curandState *states, unsigned long long seed, int n);
__global__ void monteCarloKernel(curandState *states, float *d_payoffs, float *d_deltas,
                                 float S0, float K, float r, float sigma, float T, int n);

// --- CPU BENCHMARK IMPLEMENTATION ---
void runCpuBenchmark(int N, float S0, float K, float r, float sigma, float T) {
    std::cout << "[CPU] Starting sequential benchmark (Please wait...)\n";
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    
    double sum_payoff = 0.0;
    for (int i = 0; i < N; i++) {
        float z = distribution(generator);
        float S_T = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * z);
        float payoff = fmaxf(S_T - K, 0.0f);
        sum_payoff += payoff;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    
    float price = exp(-r * T) * (sum_payoff / N);
    std::cout << "[CPU] Result: $" << price << " | Time: " << duration.count() << " ms\n";
}

int main(int argc, char **argv) {
    // 1. Configuration
    int N = 5000000;
    float S0 = 100.0f, K = 100.0f, r = 0.05f, sigma = 0.2f, T = 1.0f;
    bool run_benchmark = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "-s" && i + 1 < argc) S0 = std::stof(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) K = std::stof(argv[++i]);
        else if (arg == "--bench") run_benchmark = true;
    }

    std::cout << "==========================================\n";
    std::cout << "   CUDA FINANCIAL RISK ENGINE (v2.0)\n";
    std::cout << "==========================================\n";
    std::cout << "Simulations: " << N << "\n";

    // 2. GPU Memory Allocation
    size_t bytes = N * sizeof(float);
    float *h_payoffs = (float*)malloc(bytes);
    float *h_deltas  = (float*)malloc(bytes); 
    
    float *d_payoffs, *d_deltas;
    curandState *d_rngStates;

    nvtxRangePush("Allocation");
    CHECK_CUDA(cudaMalloc(&d_payoffs, bytes));
    CHECK_CUDA(cudaMalloc(&d_deltas, bytes)); 
    CHECK_CUDA(cudaMalloc(&d_rngStates, N * sizeof(curandState)));
    nvtxRangePop();

    // 3. GPU Initialization
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    std::cout << "[GPU] Initializing RNG... \n";
    initRNG<<<gridSize, blockSize>>>(d_rngStates, 1234ULL, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. GPU Simulation
    std::cout << "[GPU] Running Monte Carlo... \n";
    nvtxRangePush("Monte Carlo Kernel");
    auto start = std::chrono::high_resolution_clock::now();
    
    monteCarloKernel<<<gridSize, blockSize>>>(d_rngStates, d_payoffs, d_deltas, S0, K, r, sigma, T, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    std::cout << "[GPU] Copying Data... \n";
    CHECK_CUDA(cudaMemcpy(h_payoffs, d_payoffs, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_deltas, d_deltas, bytes, cudaMemcpyDeviceToHost)); 

    // 5. Reduction & Analysis
    double sum_payoff = 0.0;
    double sum_delta = 0.0;
    
    for (int i = 0; i < N; i++) {
        sum_payoff += h_payoffs[i];
        sum_delta += h_deltas[i];
    }
    
    double mean_payoff = sum_payoff / N;
    double mean_delta  = sum_delta / N;
    double option_price = exp(-r * T) * mean_payoff;
    
    // Sort for VaR (Standard C++ Sort)
    std::sort(h_payoffs, h_payoffs + N);
    float var_95 = h_payoffs[(int)(N * 0.95)];

    std::chrono::duration<float, std::milli> duration = end - start;
    
    std::cout << "\n--- RESULTS ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Option Price:      $" << option_price << "\n";
    std::cout << "Hedging Delta:     " << mean_delta << " (Sensitivity to Stock Price)\n";
    std::cout << "95% Value at Risk: $" << var_95 << "\n";
    std::cout << "GPU Time:          " << duration.count() << " ms\n";

    // 6. CPU Benchmark (Optional)
    if (run_benchmark) {
        std::cout << "\n";
        runCpuBenchmark(N, S0, K, r, sigma, T);
    }

    // 7. Output CSV (RESTORED)
    // We write the sorted tails to the CSV for visualization
    std::ofstream csv("data/risk_engine_results.csv");
    if (csv.is_open()) {
        csv << "SimulationID,SortedPayoff\n";
        int limit = (N < 1000) ? N : 1000;
        
        // Write first 1000 (lowest)
        for(int i=0; i<limit; i++) 
            csv << i << "," << h_payoffs[i] << "\n";
            
        // Write last 1000 (highest)
        for(int i=0; i<limit; i++) 
            csv << (N-limit+i) << "," << h_payoffs[N-limit+i] << "\n";
            
        csv.close();
        std::cout << "\n[IO] CSV Log saved to data/risk_engine_results.csv\n";
    } else {
        std::cerr << "\n[IO] Error: Unable to write to data/risk_engine_results.csv\n";
    }

    // Cleanup
    cudaFree(d_payoffs);
    cudaFree(d_deltas);
    cudaFree(d_rngStates);
    free(h_payoffs);
    free(h_deltas);

    return 0;
}