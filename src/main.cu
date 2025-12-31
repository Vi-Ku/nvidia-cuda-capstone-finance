#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// CUDA & Libraries
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include "common.h"

// Forward declare kernels (defined in kernels.cu)
__global__ void initRNG(curandState *states, unsigned long long seed, int n);
__global__ void monteCarloKernel(curandState *states, float *d_payoffs, 
                                 float S0, float K, float r, float sigma, float T, int n);

int main(int argc, char **argv) {
    // 1. Financial Parameters
    int N = 5000000;      // 5 Million Paths (Increased scale)
    float S0 = 100.0f;    // Stock Price
    float K = 100.0f;     // Strike Price
    float r = 0.05f;      // Risk-free Rate
    float sigma = 0.2f;   // Volatility
    float T = 1.0f;       // Time (1 year)
    
    // Parse arguments (Simplified for brevity)
    if(argc > 1) N = std::stoi(argv[1]);

    printf("--- CUDA Financial Risk Engine ---\n");
    printf("Simulations: %d\n", N);

    // 2. Memory Allocation (Using Thrust Device Vectors)
    // Thrust handles cudaMalloc/cudaFree automatically!
    nvtxRangePush("Allocation");
    thrust::device_vector<float> d_payoffs(N);
    thrust::device_vector<curandState> d_rngStates(N);
    nvtxRangePop();

    // 3. Raw Pointer Extraction
    // We need raw pointers to pass to our custom 'monteCarloKernel'
    float* raw_payoffs = thrust::raw_pointer_cast(d_payoffs.data());
    curandState* raw_states = thrust::raw_pointer_cast(d_rngStates.data());

    // 4. Initialization (Custom Kernel)
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Initializing RNG...\n");
    initRNG<<<gridSize, blockSize>>>(raw_states, 1234ULL, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Monte Carlo Simulation (Custom Kernel)
    printf("Running Monte Carlo Simulation...\n");
    nvtxRangePush("Monte Carlo Kernel");
    auto start = std::chrono::high_resolution_clock::now();
    
    monteCarloKernel<<<gridSize, blockSize>>>(raw_states, raw_payoffs, S0, K, r, sigma, T, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    // 6. Advanced Analysis using THRUST (The "More Library" part)
    
    // A. Expected Value (Reduction)
    // No copying 10M floats to CPU! We sum them on the GPU.
    nvtxRangePush("Thrust Reduction");
    float total_payoff = thrust::reduce(d_payoffs.begin(), d_payoffs.end(), 0.0f, thrust::plus<float>());
    float mean_payoff = total_payoff / N;
    float option_price = exp(-r * T) * mean_payoff;
    nvtxRangePop();

    // B. Value at Risk (Sorting)
    // To find the 95% VaR, we sort the outcomes and look at the 5th percentile.
    nvtxRangePush("Thrust Sort");
    thrust::sort(d_payoffs.begin(), d_payoffs.end());
    nvtxRangePop();

    // Extract the 95th percentile (Risk Metric)
    // We copy just ONE value back to CPU
    int index_95 = (int)(N * 0.95);
    float var_95 = d_payoffs[index_95]; 

    // 7. Reporting
    std::chrono::duration<float, std::milli> duration = end - start;
    printf("\n>>> RESULTS <<<\n");
    printf("Fair Option Price: $%.4f\n", option_price);
    printf("95%% Value at Risk: $%.4f (The payoff exceeds this in top 5%% of cases)\n", var_95);
    printf("Calculation Time:  %.2f ms\n", duration.count());
    printf("Throughput:        %.2f GPaths/s\n", (N/1e9)/(duration.count()/1000.0f));

    // 8. Dump Data for Visualization (Thrust Copy)
    // We only dump a small sample to avoid huge I/O
    std::ofstream csvFile("data/risk_engine_results.csv");
    csvFile << "SimulationID,SortedPayoff\n";
    
    // Copy first 1000 (lowest payoffs) and last 1000 (highest payoffs) to host
    int sample_size = 1000;
    thrust::host_vector<float> h_sample_low(sample_size);
    thrust::host_vector<float> h_sample_high(sample_size);
    
    thrust::copy(d_payoffs.begin(), d_payoffs.begin() + sample_size, h_sample_low.begin());
    thrust::copy(d_payoffs.end() - sample_size, d_payoffs.end(), h_sample_high.begin());

    for(int i=0; i<sample_size; i++) csvFile << i << "," << h_sample_low[i] << "\n";
    for(int i=0; i<sample_size; i++) csvFile << (N-sample_size+i) << "," << h_sample_high[i] << "\n";
    
    csvFile.close();
    printf("\nLog: Sampled tail distributions written to data/risk_engine_results.csv\n");

    return 0;
}