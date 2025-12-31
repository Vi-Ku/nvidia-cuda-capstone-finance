/*
 * CUDA Financial Risk Engine (Safe Version)
 * * Implements Monte Carlo Option Pricing using Standard CUDA C++.
 * * Removes Thrust dependency to resolve GCC 11 / NVCC header conflicts.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>

// CUDA & Libraries
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/common.h"

// Forward Declaration of Kernels
__global__ void initRNG(curandState *states, unsigned long long seed, int n);
__global__ void monteCarloKernel(curandState *states, float *d_payoffs, 
                                 float S0, float K, float r, float sigma, float T, int n);

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " -n <simulations> -s <stock> -k <strike>\n";
}

int main(int argc, char **argv) {
    // 1. Configuration
    int N = 5000000;      // 5 Million simulations
    float S0 = 100.0f;    // Stock Price
    float K = 100.0f;     // Strike Price
    float r = 0.05f;      // Risk-free Rate
    float sigma = 0.2f;   // Volatility
    float T = 1.0f;       // Time (1 Year)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "-s" && i + 1 < argc) S0 = std::stof(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) K = std::stof(argv[++i]);
    }

    std::cout << "==========================================\n";
    std::cout << "   CUDA FINANCIAL RISK ENGINE (Safe Mode)\n";
    std::cout << "==========================================\n";
    std::cout << "Simulations: " << N << "\n";

    // 2. Memory Allocation
    size_t bytes_payoffs = N * sizeof(float);
    size_t bytes_states = N * sizeof(curandState);

    float *h_payoffs = (float*)malloc(bytes_payoffs);
    float *d_payoffs;
    curandState *d_rngStates;

    // Use NVTX for profiling
    nvtxRangePush("Allocation");
    CHECK_CUDA(cudaMalloc(&d_payoffs, bytes_payoffs));
    CHECK_CUDA(cudaMalloc(&d_rngStates, bytes_states));
    nvtxRangePop();

    // 3. Initialization
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    std::cout << "[1/4] Initializing RNG... ";
    std::cout.flush();
    initRNG<<<gridSize, blockSize>>>(d_rngStates, 1234ULL, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Done.\n";

    // 4. Simulation
    std::cout << "[2/4] Running Simulations... ";
    std::cout.flush();
    
    nvtxRangePush("Monte Carlo Kernel");
    auto start = std::chrono::high_resolution_clock::now();
    
    monteCarloKernel<<<gridSize, blockSize>>>(d_rngStates, d_payoffs, S0, K, r, sigma, T, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();
    std::cout << "Done.\n";

    // 5. Data Retrieval
    std::cout << "[3/4] Copying Data (Device -> Host)... ";
    std::cout.flush();
    CHECK_CUDA(cudaMemcpy(h_payoffs, d_payoffs, bytes_payoffs, cudaMemcpyDeviceToHost));
    std::cout << "Done.\n";

    // 6. CPU Reduction & Analysis
    // We do this on CPU to avoid the Thrust header conflict
    std::cout << "[4/4] Analyzing Risk... ";
    std::cout.flush();
    
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += h_payoffs[i];
    }
    double mean_payoff = sum / N;
    double option_price = exp(-r * T) * mean_payoff;

    // Sort on CPU for VaR (Standard C++ Sort)
    std::sort(h_payoffs, h_payoffs + N);
    int index_95 = (int)(N * 0.95);
    float var_95 = h_payoffs[index_95];
    std::cout << "Done.\n";

    // 7. Reporting
    std::chrono::duration<float, std::milli> duration = end - start;
    float milliseconds = duration.count();
    float gpaths = (N / 1e9f) / (milliseconds / 1000.0f);

    std::cout << "\n------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Fair Option Price:   $" << option_price << "\n";
    std::cout << "95% Value at Risk:   $" << var_95 << "\n";
    std::cout << "Calculation Time:    " << milliseconds << " ms\n";
    std::cout << "Throughput:          " << gpaths << " GPaths/s\n";
    std::cout << "------------------------------------------\n";

    // 8. Output CSV
    std::ofstream csv("data/risk_engine_results.csv");
    csv << "SimulationID,SortedPayoff\n";
    // Save tails
    for(int i=0; i<1000; i++) csv << i << "," << h_payoffs[i] << "\n";
    for(int i=0; i<1000; i++) csv << (N-1000+i) << "," << h_payoffs[N-1000+i] << "\n";
    csv.close();
    std::cout << "Log: Data written to data/risk_engine_results.csv\n";

    // Cleanup
    cudaFree(d_payoffs);
    cudaFree(d_rngStates);
    free(h_payoffs);

    return 0;
}