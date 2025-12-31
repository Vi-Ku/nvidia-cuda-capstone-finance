# nvidia-cuda-capstone-finance
CUDA-Accelerated Monte Carlo Option Pricing

## 1. Project Overview

This project implements a high-performance **Monte Carlo Option Pricing Engine** using **CUDA C++**. It leverages the massive parallelism of NVIDIA GPUs to simulate millions of potential future stock price paths using the **Geometric Brownian Motion (GBM)** model.

By offloading the heavy statistical computations to the GPU, this application achieves a significant speedup compared to sequential CPU implementations, making it suitable for real-time financial risk analysis and derivative pricing.

## 2. Problem Statement

In quantitative finance, determining the fair value of "European Call Options" is a computationally intensive task. The **Black-Scholes model** provides a closed-form solution, but complex derivatives often require numerical methods like Monte Carlo simulations.

To obtain a statistically significant "Confidence Interval" for the price, analysts must simulate **tens of millions** of random market scenarios.

* **The Challenge:** Running 10 million simulations on a standard CPU is slow and inefficient for real-time trading desks.
* **The Goal:** Calculate the fair price of an option and analyze the risk distribution (profit/loss) in milliseconds.

## 3. Our Approach

We utilize **Parallel Computing** to treat every potential market scenario as an independent thread.

1. **Parallel Simulation:** We launch millions of CUDA threads. Each thread simulates a single "path" of the stock price over time `T`.
2. **RNG on GPU:** We use the **NVIDIA cuRAND** library to generate high-quality pseudo-random numbers directly on the device, avoiding the bottleneck of transferring random numbers from Host to Device.
3. **Payoff Calculation:** Each thread independently calculates the "Call Option Payoff" () for its specific path.
4. **Reduction:** We aggregate the results to find the expected (mean) payoff and discount it to the present value.

## 3. Advanced Features & Library Usage
This project goes beyond basic kernels by integrating the **Thrust Library** to perform "Data-Parallel Primitives" directly on the GPU.

### 🚀 **Features**
1.  **Hybrid Architecture:** Combines custom `__global__` kernels (for complex GBM math) with `Thrust` algorithms (for data reduction).
2.  **Zero-Copy Reduction:** Uses `thrust::reduce` to calculate the average option price entirely in GPU memory, avoiding the bottleneck of transferring 50MB+ of simulation data over the PCIe bus.
3.  **Value at Risk (VaR):** Implements `thrust::sort` (Radix Sort) to order millions of outcomes and pinpoint the 95th percentile risk metric in milliseconds.
4.  **NVTX Profiling:** Annotated with `nvtxRangePush/Pop` to visualize the interplay between Kernel execution, Thrust allocation, and Sorting phases in Nsight Systems.

### 📚 **Libraries Used**
* **cuRAND:** High-quality parallel pseudo-random number generation (XORWOW algorithm).
* **Thrust:** C++ template library for CUDA. Used for `device_vector` memory management, `reduce` (Summation), and `sort` (Ranking).

## 4. Technical Implementation

* **`src/kernels.cu`**: Contains the core GPU kernels.
* `initRNG`: Initializes the `curandState` for each thread.
* `monteCarloKernel`: Performs the GBM path simulation and payoff calculation.


* **`include/common.h`**: Defines standard error-checking macros (`CHECK_CUDA`) and NVTX profiling markers.
* **`data/plot_distribution.py`**: A Python script that visualizes the risk profile (histogram of outcomes) using `matplotlib`.
* **Profiling**: The application uses **NVTX (NVIDIA Tools Extension)** ranges, allowing developers to visualize the "Initialization" vs. "Computation" phases in NVIDIA Nsight Systems.

## 5. Dependencies

To build and run this project, you need:

* **NVIDIA CUDA Toolkit** (11.0 or higher)
* Must have `nvcc` in your PATH.
* Links against `libcurand` and `libnvToolsExt`.


* **Make** (for building the project)
* **Python 3** (for visualization)
* Required libraries: `pandas`, `matplotlib`
* Install via: `pip install pandas matplotlib`



## 6. Directory Structure

```
cuda-monte-carlo-pricing/
├── bin/            # Compiled executable (option_pricer)
├── data/           # Output CSV logs and PNG plots
├── docs/           # Project documentation
├── include/        # Header files (common.h)
├── src/            # Source code (.cu files)
├── Makefile        # Build script
├── README.md       # Project documentation
└── run.sh          # Automated execution script

```

## 7. How to Use

### Compilation

Navigate to the root directory and run `make`. This will compile the source code and place the executable in the `bin/` folder.

```bash
make

```

### Execution

You can run the pricer manually with custom financial parameters:

```bash
./bin/option_pricer -n <simulations> -s <stock_price> -k <strike_price>

```

* `-n`: Number of paths to simulate (e.g., 1000000)
* `-s`: Initial Stock Price ()
* `-k`: Strike Price ()

### Automated Run (Recommended)

Use the provided shell script to build, run both Small and Large scale tests, and generate the visualization automatically:

```bash
./run.sh

```

## 8. Proof of Execution

After running the project, check the `data/` folder for artifacts:

1. **`simulation_results.csv`**: A log of the payoff calculated for the first 1,000 paths.
2. **`risk_profile.png`**: A histogram showing the probability distribution of profitable outcomes, generated by the Python script.

This visual output demonstrates that the simulation followed a Lognormal distribution as expected by financial theory.