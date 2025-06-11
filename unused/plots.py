import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import combinations

# Load the metrics results
df = pd.DataFrame()
try:
    df = pd.read_csv("metrics_results.csv")
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"Error loading the CSV file: {e}")
    exit(1)

# Remove the ID column if it exists
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Get all metric columns
metric_columns = df.columns.tolist()
print(f"Metrics to plot: {metric_columns}")

# Calculate how many pairs we'll have
metric_pairs = list(combinations(metric_columns, 2))
num_pairs = len(metric_pairs)
print(f"Total number of metric pairs to plot: {num_pairs}")

# Define the grid layout
plots_per_figure = 16  # 4x4 grid
num_figures = math.ceil(num_pairs / plots_per_figure)

# Create figures with subplots
for fig_num in range(num_figures):
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(f'Metric Correlations (Figure {fig_num+1} of {num_figures})', fontsize=16)
    
    axes = axes.flatten()
    
    # Get the pairs for this figure
    start_idx = fig_num * plots_per_figure
    end_idx = min(start_idx + plots_per_figure, num_pairs)
    current_pairs = metric_pairs[start_idx:end_idx]
    
    for i, (metric1, metric2) in enumerate(current_pairs):
        ax = axes[i]
        
        # Create a temporary dataframe with just the two metrics we're plotting
        temp_df = df[[metric1, metric2]].copy()
        
        # Sort by the x-axis metric
        temp_df = temp_df.sort_values(by=metric1)
        
        # Create scatter plot with connected dots (sorted by x-axis)
        ax.plot(temp_df[metric1], temp_df[metric2], 'o-', markersize=4, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(metric1)
        ax.set_ylabel(metric2)
        ax.set_title(f'{metric1} vs {metric2}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
    # Hide unused subplots
    for j in range(len(current_pairs), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title
    plt.savefig(f'plots/metric_correlations_fig{fig_num+1}.png', dpi=300)
    print(f"Saved figure {fig_num+1}")

# Create a correlation heatmap as a summary
plt.figure(figsize=(14, 12))
corr = df.corr()
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

# Add correlation values
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                 ha='center', va='center', 
                 color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black',
                 fontsize=8)

plt.title('Correlation Heatmap of All Metrics')
plt.tight_layout()
plt.savefig('plots/metrics_correlation_heatmap.png', dpi=300)
print("Saved correlation heatmap")

print("All plots have been generated successfully!")
