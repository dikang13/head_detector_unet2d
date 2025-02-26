import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_training_curves(csv_path, output_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Find best validation metrics
    best_val_loss = df['val_loss'].min()
    best_val_metric = df['val_metric'].max()
    best_val_loss_epoch = df['val_loss'].idxmin()
    best_val_metric_epoch = df['val_metric'].idxmax()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
    
    # Increase font sizes
    plt.rcParams.update({'font.size': 14})
    title_fontsize = 18
    legend_fontsize = 16
    
    # Plot 1: Loss curves
    ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.axvline(x=best_val_loss_epoch, color='g', linestyle='--', linewidth=2, alpha=0.5, 
                label=f'Best Val Loss (Epoch {best_val_loss_epoch})')
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.set_title(f'UNet Training: Best Val Loss = {best_val_loss:.4f} (Epoch {best_val_loss_epoch})', 
                 fontsize=title_fontsize, fontweight='bold')
    ax1.legend(fontsize=legend_fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: Metric curves
    ax2.plot(df['epoch'], df['val_metric'], 'r-', linewidth=2, label='Validation Metric')
    ax2.axvline(x=best_val_metric_epoch, color='g', linestyle='--', linewidth=2, alpha=0.5, 
                label=f'Best Val Metric (Epoch {best_val_metric_epoch})')
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.set_ylabel('Metric', fontsize=16)
    ax2.set_ylim([0.3, 0.9])  # Fixed y-limits for metric subplot
    ax2.set_title(f'UNet Performance: Best Val Metric = {best_val_metric:.4f} (Epoch {best_val_metric_epoch})', 
                 fontsize=title_fontsize, fontweight='bold')
    ax2.legend(fontsize=legend_fontsize)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Improve spacing and layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Plot UNet training curves from CSV file')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file containing training data')
    parser.add_argument('--output_path', type=str, help='Path where the output PNG will be saved')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate plot
    plot_training_curves(args.csv_path, args.output_path)

if __name__ == "__main__":
    main()