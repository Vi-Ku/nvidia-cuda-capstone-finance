import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_risk_profile(csv_file, output_image):
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return

    try:
        df = pd.read_csv(csv_file)
        
        # Handle column naming differences
        col_name = 'SortedPayoff' if 'SortedPayoff' in df.columns else 'Payoff'
        
        plt.figure(figsize=(10, 6))
        
        # Plot Histogram
        # We use a fixed range (0 to 100) to make comparing the 3 images easier
        plt.hist(df[col_name], bins=50, color='royalblue', edgecolor='black', alpha=0.7)
        
        plt.title(f'Risk Profile: {os.path.basename(output_image)}')
        plt.xlabel('Option Payoff ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_image)
        print(f"Success: Saved plot to {output_image}")
        plt.close() # Close memory to prevent leaks
        
    except Exception as e:
        print(f"Error plotting {csv_file}: {e}")

if __name__ == "__main__":
    # Default values if no arguments provided
    input_csv = "data/risk_engine_results.csv"
    output_png = "data/risk_profile.png"
    
    # Override with command line arguments if they exist
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_png = sys.argv[2]
        
    plot_risk_profile(input_csv, output_png)