#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h> 

// Professional Error Checking Macro
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#endif