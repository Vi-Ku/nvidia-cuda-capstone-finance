#!/bin/bash

# =========================================================
# CUDA Financial Risk Engine - Multi-Scenario Analysis
# =========================================================

mkdir -p data bin

# 1. Compile
echo "[1/5] Compiling..."
make > /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

# 2. SCENARIO 1: NEUTRAL (Stock $100 vs Strike $100)
echo ""
echo "--- SCENARIO 1: NEUTRAL MARKET ---"
./bin/risk_engine -n 1000000 -s 100 -k 100
mv data/risk_engine_results.csv data/results_neutral.csv

# 3. SCENARIO 2: BULL MARKET (Stock $120 vs Strike $100)
# Expect high payoffs
echo ""
echo "--- SCENARIO 2: BULL MARKET ---"
./bin/risk_engine -n 1000000 -s 120 -k 100
mv data/risk_engine_results.csv data/results_bull.csv

# 4. SCENARIO 3: BEAR MARKET (Stock $80 vs Strike $100)
# Expect mostly zero payoffs (Options expire worthless)
echo ""
echo "--- SCENARIO 3: BEAR MARKET ---"
./bin/risk_engine -n 1000000 -s 80 -k 100
mv data/risk_engine_results.csv data/results_bear.csv

# 5. Generate ALL Visualizations
echo ""
echo "[5/5] Generating Plots for all scenarios..."

# Plot Neutral
python3 data/plot_results.py data/results_neutral.csv data/plot_neutral.png

# Plot Bull
python3 data/plot_results.py data/results_bull.csv data/plot_bull.png

# Plot Bear
python3 data/plot_results.py data/results_bear.csv data/plot_bear.png

echo ""
echo "Done! Generated 3 artifacts in data/:"
echo "1. data/plot_neutral.png (Standard Curve)"
echo "2. data/plot_bull.png    (Shifted Right -> High Profit)"
echo "3. data/plot_bear.png    (Shifted Left  -> Mostly Zero)"