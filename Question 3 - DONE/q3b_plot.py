import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Set font sizes for better readability
plt.rcParams.update({'font.size': 12})
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

# Read the results file
results_file = "puf_model_scaling_results.csv"

if not os.path.exists(results_file):
    sys.exit(0)  # Exit silently if file not found

try:
    # Load the data
    df = pd.read_csv(results_file)
    
    # Sort by sample size for proper plotting
    df = df.sort_values('sample_size')
    
    # Convert sample size to percentage for better readability
    df['sample_size_percent'] = df['sample_size'] * 100
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Execution Time (standard scale)
    ax1 = axes[0, 0]
    ax1.plot(df['sample_size_percent'], df['rf_time'], 'o-', color='blue', label='Random Forest')
    ax1.plot(df['sample_size_percent'], df['gbt_time'], 's-', color='green', label='Gradient Boosting')
    ax1.plot(df['sample_size_percent'], df['mlp_time'], '^-', color='red', label='MLP')
    ax1.set_xlabel('Sample Size (%)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Model Training Time vs Sample Size')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot 2: Execution Time (log scale)
    ax2 = axes[0, 1]
    ax2.plot(df['sample_size_percent'], df['rf_time'], 'o-', color='blue', label='Random Forest')
    ax2.plot(df['sample_size_percent'], df['gbt_time'], 's-', color='green', label='Gradient Boosting')
    ax2.plot(df['sample_size_percent'], df['mlp_time'], '^-', color='red', label='MLP')
    ax2.set_xlabel('Sample Size (%)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Model Training Time vs Sample Size (Log Scale)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yscale('log')
    ax2.legend()
    
    # Plot 3: Accuracy - Make sure we're plotting the correct columns
    ax3 = axes[1, 0]
    
    # Check if columns exist in the dataframe
    if 'rf_accuracy' in df.columns and 'gbt_accuracy' in df.columns and 'mlp_accuracy' in df.columns:
        ax3.plot(df['sample_size_percent'], df['rf_accuracy'], 'o-', color='blue', label='Random Forest')
        ax3.plot(df['sample_size_percent'], df['gbt_accuracy'], 's-', color='green', label='Gradient Boosting')
        ax3.plot(df['sample_size_percent'], df['mlp_accuracy'], '^-', color='red', label='MLP')
        
        # Only set ylim if there's actual data to plot
        min_acc = min(df['rf_accuracy'].min(), df['gbt_accuracy'].min(), df['mlp_accuracy'].min())
        max_acc = max(df['rf_accuracy'].max(), df['gbt_accuracy'].max(), df['mlp_accuracy'].max())
        if max_acc > min_acc:
            # Add a small buffer
            buffer = (max_acc - min_acc) * 0.1
            ax3.set_ylim([max(0, min_acc - buffer), min(1.0, max_acc + buffer)])
        else:
            # Default range if all values are the same
            ax3.set_ylim([max(0, min_acc - 0.05), min(1.0, min_acc + 0.05)])
    else:
        # If columns don't exist, try alternative names
        possible_accuracy_cols = [col for col in df.columns if 'acc' in col.lower()]
        for col in possible_accuracy_cols:
            if 'rf' in col.lower() or 'random' in col.lower():
                ax3.plot(df['sample_size_percent'], df[col], 'o-', color='blue', label='Random Forest')
            elif 'gbt' in col.lower() or 'boost' in col.lower():
                ax3.plot(df['sample_size_percent'], df[col], 's-', color='green', label='Gradient Boosting')
            elif 'mlp' in col.lower() or 'perceptron' in col.lower():
                ax3.plot(df['sample_size_percent'], df[col], '^-', color='red', label='MLP')
            
    ax3.set_xlabel('Sample Size (%)')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Model Accuracy vs Sample Size')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Plot 4: AUC - Make sure we're plotting the correct columns
    ax4 = axes[1, 1]
    
    # Check if columns exist in the dataframe
    if 'rf_auc' in df.columns and 'gbt_auc' in df.columns and 'mlp_auc' in df.columns:
        ax4.plot(df['sample_size_percent'], df['rf_auc'], 'o-', color='blue', label='Random Forest')
        ax4.plot(df['sample_size_percent'], df['gbt_auc'], 's-', color='green', label='Gradient Boosting')
        ax4.plot(df['sample_size_percent'], df['mlp_auc'], '^-', color='red', label='MLP')
        
        # Only set ylim if there's actual data to plot
        min_auc = min(df['rf_auc'].min(), df['gbt_auc'].min(), df['mlp_auc'].min())
        max_auc = max(df['rf_auc'].max(), df['gbt_auc'].max(), df['mlp_auc'].max())
        if max_auc > min_auc:
            # Add a small buffer
            buffer = (max_auc - min_auc) * 0.1
            ax4.set_ylim([max(0, min_auc - buffer), min(1.0, max_auc + buffer)])
        else:
            # Default range if all values are the same
            ax4.set_ylim([max(0, min_auc - 0.05), min(1.0, min_auc + 0.05)])
    else:
        # If columns don't exist, try alternative names
        possible_auc_cols = [col for col in df.columns if 'auc' in col.lower() or 'roc' in col.lower()]
        for col in possible_auc_cols:
            if 'rf' in col.lower() or 'random' in col.lower():
                ax4.plot(df['sample_size_percent'], df[col], 'o-', color='blue', label='Random Forest')
            elif 'gbt' in col.lower() or 'boost' in col.lower():
                ax4.plot(df['sample_size_percent'], df[col], 's-', color='green', label='Gradient Boosting')
            elif 'mlp' in col.lower() or 'perceptron' in col.lower():
                ax4.plot(df['sample_size_percent'], df[col], '^-', color='red', label='MLP')
    
    ax4.set_xlabel('Sample Size (%)')
    ax4.set_ylabel('AUC')
    ax4.set_title('Model AUC vs Sample Size')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend()
    
    # Add a main title
    plt.suptitle('PUF Classification Model Performance Across Sample Sizes', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust to make room for suptitle
    
    # Save figure
    plt.savefig('puf_model_scaling_results.png', dpi=300, bbox_inches='tight')
    
    # Also create a combined plot to compare time vs performance
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Make sure we have the correct columns before plotting
    if all(col in df.columns for col in ['rf_time', 'gbt_time', 'mlp_time', 'rf_accuracy', 'gbt_accuracy', 'mlp_accuracy']):
        # Execution time vs. Accuracy
        ax1.plot(df['rf_time'], df['rf_accuracy'], 'o-', color='blue', label='Random Forest')
        ax1.plot(df['gbt_time'], df['gbt_accuracy'], 's-', color='green', label='Gradient Boosting')
        ax1.plot(df['mlp_time'], df['mlp_accuracy'], '^-', color='red', label='MLP')
        
        # Only set ylim if there's actual data to plot
        min_acc = min(df['rf_accuracy'].min(), df['gbt_accuracy'].min(), df['mlp_accuracy'].min())
        max_acc = max(df['rf_accuracy'].max(), df['gbt_accuracy'].max(), df['mlp_accuracy'].max())
        if max_acc > min_acc:
            # Add a small buffer
            buffer = (max_acc - min_acc) * 0.1
            ax1.set_ylim([max(0, min_acc - buffer), min(1.0, max_acc + buffer)])
    
    ax1.set_xlabel('Execution Time (seconds)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy vs. Execution Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Make sure we have the correct columns before plotting
    if all(col in df.columns for col in ['rf_time', 'gbt_time', 'mlp_time', 'rf_auc', 'gbt_auc', 'mlp_auc']):
        # Execution time vs. AUC
        ax2.plot(df['rf_time'], df['rf_auc'], 'o-', color='blue', label='Random Forest')
        ax2.plot(df['gbt_time'], df['gbt_auc'], 's-', color='green', label='Gradient Boosting')
        ax2.plot(df['mlp_time'], df['mlp_auc'], '^-', color='red', label='MLP')
        
        # Only set ylim if there's actual data to plot
        min_auc = min(df['rf_auc'].min(), df['gbt_auc'].min(), df['mlp_auc'].min())
        max_auc = max(df['rf_auc'].max(), df['gbt_auc'].max(), df['mlp_auc'].max())
        if max_auc > min_auc:
            # Add a small buffer
            buffer = (max_auc - min_auc) * 0.1
            ax2.set_ylim([max(0, min_auc - buffer), min(1.0, max_auc + buffer)])
    
    ax2.set_xlabel('Execution Time (seconds)')
    ax2.set_ylabel('AUC')
    ax2.set_title('Model AUC vs. Execution Time')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.suptitle('Performance vs. Computational Cost Trade-off', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig('puf_model_tradeoff_results.png', dpi=300, bbox_inches='tight')
    
    # Save plots without showing or printing stats
    plt.close(fig)
    plt.close(fig2)
    
except Exception:
    sys.exit(0)  # Exit silently on error