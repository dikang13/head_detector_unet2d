import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_tensorboard_data(log_dir):
    """Load loss data from TensorBoard logs."""
    train_losses = []
    val_losses = []
    steps = []
    
    # Get all event files sorted by modification time
    event_files = tf.io.gfile.glob(f"{log_dir}/events*")
    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir}")
        
    print(f"Found {len(event_files)} event files:")
    for f in event_files:
        mtime = os.path.getmtime(f)
        print(f"- {os.path.basename(f)} (modified: {mtime})")
    
    # Process each event file
    for event_file in sorted(event_files, key=os.path.getmtime):
        try:
            for e in tf.compat.v1.train.summary_iterator(event_file):
                for v in e.summary.value:
                    if v.tag == 'Loss/train':
                        train_losses.append((e.step, v.simple_value))
                    elif v.tag == 'Loss/val':
                        val_losses.append((e.step, v.simple_value))
        except tf.errors.DataLossError as ex:
            print(f"Warning: Corrupt event file {os.path.basename(event_file)}: {ex}")
            continue
    
    # Sort by steps and separate into x and y values
    train_losses.sort(key=lambda x: x[0])
    val_losses.sort(key=lambda x: x[0])
    
    train_steps, train_values = zip(*train_losses) if train_losses else ([], [])
    val_steps, val_values = zip(*val_losses) if val_losses else ([], [])
    
    return train_steps, train_values, val_steps, val_values

def plot_losses(train_steps, train_values, val_steps, val_values, save_path=None):
    """Create a plot of training and validation losses."""
    plt.figure(figsize=(12, 6))
    
    if train_values:
        plt.plot(train_steps, train_values, 'b-', label='Training Loss', alpha=0.7)
    if val_values:
        plt.plot(val_steps, val_values, 'r-', label='Validation Loss', alpha=0.7)
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add text box with final values
    if train_values and val_values:
        final_train = train_values[-1]
        final_val = val_values[-1]
        best_val = min(val_values)
        plt.text(0.02, 0.98, 
                f'Final train loss: {final_train:.4f}\n'
                f'Final val loss: {final_val:.4f}\n'
                f'Best val loss: {best_val:.4f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot losses from TensorBoard logs')
    parser.add_argument('--logdir', type=str, required=True,
                       help='Path to TensorBoard log directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the output plot. If not specified, saves in the log directory.')
    
    args = parser.parse_args()
    
    # If no output path specified, save in the log directory
    if args.output is None:
        # Extract the experiment name from the log directory path
        exp_name = os.path.basename(args.logdir.rstrip('/'))
        args.output = os.path.join(args.logdir, f'{exp_name}_losses.png')
    
    train_steps, train_values, val_steps, val_values = load_tensorboard_data(args.logdir)
    plot_losses(train_steps, train_values, val_steps, val_values, args.output)

if __name__ == '__main__':
    main()