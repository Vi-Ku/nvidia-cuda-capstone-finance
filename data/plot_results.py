import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_risk_profile(csv_file):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Run the simulation first.")
        return

    try:
        df = pd.read_csv(csv_file)
        
        # We plotted the "tails" (lowest and highest outcomes) in the CSV
        # Let's plot the distribution of these extremes
        plt.figure(figsize=(10, 6))
        
        plt.hist(df['SortedPayoff'], bins=50, color='royalblue', edgecolor='black', alpha=0.7)
        
        plt.title('Tail Risk Distribution (Extreme Outcomes)')
        plt.xlabel('Option Payoff ($)')
        plt.ylabel('Frequency (in sample)')
        plt.grid(True, alpha=0.3)
        
        output_file = 'data/risk_profile.png'
        plt.savefig(output_file)
        print(f"Success: Risk profile saved to {output_file}")
        
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    plot_risk_profile("data/risk_engine_results.csv")