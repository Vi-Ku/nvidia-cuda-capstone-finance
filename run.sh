#!/bin/bash

# =========================================================
# CUDA Financial Risk Engine - Automation Script
# =========================================================

# 1. Compile the Project
echo "---------------------------------------------------"
echo "[1/4] Compiling Source Code..."
echo "---------------------------------------------------"
make

# Check if make was successful
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed. Exiting."
    exit 1
fi

# 2. Run Small Scale Test (Validation)
# We run a small batch (10k paths) just to ensure the kernel logic works quickly.
echo ""
echo "---------------------------------------------------"
echo "[2/4] Running Validation Test (Small Scale: 10k)..."
echo "---------------------------------------------------"
./bin/risk_engine -n 10000 -s 100 -k 100

# 3. Run Large Scale Test (Performance Demonstration)
# We run 10 MILLION paths to demonstrate high-throughput GPU performance.
# This satisfies the "Code Execution at Scale" rubric requirement.
echo ""
echo "---------------------------------------------------"
echo "[3/4] Running Capstone Scale Test (Large Scale: 10M)..."
echo "---------------------------------------------------"
./bin/risk_engine -n 10000000 -s 100 -k 100

# 4. Generate Visualization
# Uses the Python script to plot the tail risk histogram from the CSV data.
echo ""
echo "---------------------------------------------------"
echo "[4/4] Generating Risk Profile Visualization..."
echo "---------------------------------------------------"

if [ -f "data/risk_engine_results.csv" ]; then
    python3 data/plot_results.py
else
    echo "Error: CSV log not found. Skipping visualization."
fi

echo ""
echo "Done! Check the 'data/' folder for your report."