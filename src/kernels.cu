#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "common.h"

// Kernel 1: Initialize RNG States
// Each thread gets a unique random seed state
__global__ void initRNG(curandState *states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel 2: Monte Carlo Simulation (Black-Scholes Path)
// Calculates the path and stores the resulting Payoff
__global__ void monteCarloKernel(curandState *states, float *d_payoffs, 
                                 float S0, float K, float r, float sigma, float T, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState localState = states[idx];
        
        // Generate Gaussian noise Z ~ N(0,1)
        float z = curand_normal(&localState);
        
        // Geometric Brownian Motion: S_T = S0 * exp(...)
        float S_T = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * z);
        
        // Call Option Payoff: max(S_T - K, 0)
        d_payoffs[idx] = fmaxf(S_T - K, 0.0f);
        
        // Update state
        states[idx] = localState;
    }
}