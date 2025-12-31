/*
 * CUDA Financial Risk Engine - Capstone Project
 * * This file orchestrates the Monte Carlo simulation using a hybrid approach:
 * 1. Custom CUDA Kernels: For efficient path simulation (Black-Scholes GBM).
 * 2. Thrust Library: For high-performance reduction (Mean) and sorting (VaR).
 * 3. NVTX Profiling: For visualizing performance in Nsight Systems.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>

// CUDA Runtime
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thrust Library (The "STL of CUDA")
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

// Project Header
#include "../include/common.h"

// Forward Declaration of Kernels (Implemented in kernels.cu)
__global__ void initRNG(curandState *states, unsigned long long seed, int n);
__global__ void monteCarloKernel(curandState *states, float *d_payoffs, 
                                 float S0, float K, float r, float sigma, float T, int n);

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  -n <int>    Number of simulations (default: 5,000,000)\n"
              << "  -s <float>  Stock price S0 (default: 100.0)\n"
              << "  -k <float>  Strike price K (default: 100.0)\n"
              << "  -h          Show help\n";
}

int main(int argc, char **argv) {
    // -------------------------------------------------------------------------
    // 1. Configuration & Argument Parsing
    // -------------------------------------------------------------------------
    int N = 5000000;      // 5 Million simulations (Scale)
    float S0 = 100.0f;    // Initial Stock Price
    float K = 100.0f;     // Strike Price
    float r = 0.05f;      // Risk-free Rate (5%)
    float sigma = 0.2f;   // Volatility (20%)
    float T = 1.0f;       // Time Horizon (1 Year)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) N = std::stoi(argv[++i]);
        else if (arg == "-s" && i + 1 < argc) S0 = std::stof(argv[++i]);
        else if (arg == "-k" && i + 1 < argc) K = std::stof(argv[++i]);
        else if (arg == "-h") { printUsage(argv[0]); return 0; }
    }

    std::cout << "==========================================\n";
    std::cout << "   CUDA FINANCIAL RISK ENGINE (Monte Carlo)\n";
    std::cout << "==========================================\n";
    std::cout << "Parameters:\n";
    std::cout << "  Simulations: " << N << "\n";
    std::cout << "  Stock (S0):  $" << S0 << "\n";
    std::cout << "  Strike (K):  $" << K << "\n";
    std::cout << "------------------------------------------\n";

    // -------------------------------------------------------------------------
    // 2. Resource Allocation (Thrust)
    // -------------------------------------------------------------------------
    // Using Thrust vectors handles cudaMalloc/cudaFree automatically.
    nvtxRangePush("Memory Allocation");
    
    // Vector to store the resulting payoff of every single simulation path
    thrust::device_vector<float> d_payoffs(N);
    
    // Vector to store the random number generator state for every thread
    thrust::device_vector<curandState> d_rngStates(N);
    
    nvtxRangePop();

    // Get raw pointers to pass to custom kernels
    float* raw_payoffs = thrust::raw_pointer_cast(d_payoffs.data());
    curandState* raw_states = thrust::raw_pointer_cast(d_rngStates.data());

    // Grid Calculation
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // -------------------------------------------------------------------------
    // 3. RNG Initialization (Custom Kernel)
    // -------------------------------------------------------------------------
    std::cout << "[1/4] Initializing RNG States... ";
    std::cout.flush();
    
    nvtxRangePush("Init RNG Kernel");
    initRNG<<<gridSize, blockSize>>>(raw_states, 1234ULL, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();
    
    std::cout << "Done.\n";

    // -------------------------------------------------------------------------
    // 4. Monte Carlo Simulation (Custom Kernel)
    // -------------------------------------------------------------------------
    std::cout << "[2/4] Running Path Simulations... ";
    std::cout.flush();

    nvtxRangePush("Monte Carlo Kernel");
    auto start = std::chrono::high_resolution_clock::now();

    monteCarloKernel<<<gridSize, blockSize>>>(raw_states, raw_payoffs, S0, K, r, sigma, T, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    std::cout << "Done.\n";

    // -------------------------------------------------------------------------
    // 5. Financial Analysis (Thrust Library)
    // -------------------------------------------------------------------------
    std::cout << "[3/4] Reducing Data (Calculate Mean)... ";
    std::cout.flush();

    // A. Expected Value (Mean Price)
    // Use thrust::reduce to sum 5 million floats on the GPU instantly
    nvtxRangePush("Thrust Reduction");
    float total_payoff = thrust::reduce(d_payoffs.begin(), d_payoffs.end(), 0.0f, thrust::plus<float>());
    nvtxRangePop();

    float mean_payoff = total_payoff / N;
    // Discount to present value: Price = e^(-rT) * E[Payoff]
    float option_price = expf(-r * T) * mean_payoff;
    std::cout << "Done.\n";

    // B. Value at Risk (VaR) Calculation
    // We need to sort the outcomes to find the 95th percentile worst case.
    // thrust::sort uses a highly optimized Radix Sort.
    std::cout << "[4/4] Sorting for Risk Analysis (VaR)... ";
    std::cout.flush();

    nvtxRangePush("Thrust Sort");
    thrust::sort(d_payoffs.begin(), d_payoffs.end());
    nvtxRangePop();

    // The 95% VaR is the value where 95% of outcomes are worse (or better, depending on definition).
    // Here we look at the top 5% of returns (Deep In-the-Money).
    int index_95 = (int)(N * 0.95);
    float var_95 = d_payoffs[index_95];
    std::cout << "Done.\n";

    // -------------------------------------------------------------------------
    // 6. Reporting & Logging
    // -------------------------------------------------------------------------
    std::chrono::duration<float, std::milli> duration = end - start;
    float milliseconds = duration.count();
    float gpaths = (N / 1e9f) / (milliseconds / 1000.0f);

    std::cout << "\n------------------------------------------\n";
    std::cout << "             FINAL RESULTS                \n";
    std::cout << "------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Fair Option Price:   $" << option_price << "\n";
    std::cout << "95% Value at Risk:   $" << var_95 << " (Top 5% Outcome)\n";
    std::cout << "Calculation Time:    " << milliseconds << " ms\n";
    std::cout << "Throughput:          " << gpaths << " GPaths/s\n";
    std::cout << "------------------------------------------\n";

    // Export Data for Visualization
    // We export the tails (lowest 1000 and highest 1000) to keep the CSV small
    std::string outFile = "data/risk_engine_results.csv";
    std::ofstream csvFile(outFile);
    
    if (csvFile.is_open()) {
        csvFile << "SimulationID,SortedPayoff\n";
        
        // Copy tails to host
        int sample_size = 1000;
        thrust::host_vector<float> h_low(sample_size);
        thrust::host_vector<float> h_high(sample_size);

        thrust::copy(d_payoffs.begin(), d_payoffs.begin() + sample_size, h_low.begin());
        thrust::copy(d_payoffs.end() - sample_size, d_payoffs.end(), h_high.begin());

        for(int i = 0; i < sample_size; i++) csvFile << i << "," << h_low[i] << "\n";
        for(int i = 0; i < sample_size; i++) csvFile << (N - sample_size + i) << "," << h_high[i] << "\n";

        std::cout << "Log: Sampled tail data written to " << outFile << "\n";
    } else {
        std::cerr << "Error: Could not open output file.\n";
    }

    return 0;
}