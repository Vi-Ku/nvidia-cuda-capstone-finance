#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/common.h"

// Kernel 1: Initialize RNG States
__global__ void initRNG(curandState *states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel 2: Monte Carlo Simulation (Price + Delta)
// We now calculate TWO things: The Payoff and the Hedging Delta
__global__ void monteCarloKernel(curandState *states, float *d_payoffs, float *d_deltas,
                                 float S0, float K, float r, float sigma, float T, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState localState = states[idx];
        
        // Generate Gaussian noise Z ~ N(0,1)
        float z = curand_normal(&localState);
        
        // Geometric Brownian Motion formula
        float S_T = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * z);
        
        // 1. Call Option Payoff: max(S_T - K, 0)
        float payoff = fmaxf(S_T - K, 0.0f);
        d_payoffs[idx] = payoff;

        // 2. Calculate "Delta" (Pathwise Estimator)
        // If the option finishes in-the-money (S_T > K), Delta contrib is (S_T / S0)
        // If out-of-the-money, Delta contrib is 0
        if (S_T > K) {
            d_deltas[idx] = expf(-r * T) * (S_T / S0);
        } else {
            d_deltas[idx] = 0.0f;
        }
        
        // Update state
        states[idx] = localState;
    }
}