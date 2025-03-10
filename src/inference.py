# inference.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet2d.model.unet_model import UNet2D
from unet2d.model.metrics import CustomDiceCoefficientWithLogits
from data_loader import load_test_data
from config import load_config
import os
import csv
import argparse
from tqdm import tqdm

def load_model(model_path, n_features, device):
    """Load the trained UNet model."""
    model = UNet2D(1, 1, n_features, False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
    

def save_prediction(image, truth, pred, save_path, index, dice_score=None):
    """Save the prediction visualization."""
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image[0].squeeze(), cmap='gray')  # Remove channel dimension
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth
    plt.subplot(132)
    plt.imshow(truth[0].squeeze(), cmap='jet')  # Remove channel dimension
    title = 'Ground truth'
    if dice_score is not None:
        title += f' (Dice: {dice_score:.3f})'
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    
    # Prediction
    plt.subplot(133)
    if dice_score is not None:
        plt.imshow(pred[0].squeeze(), cmap='jet')  # Remove channel dimension
        plt.title('Prediction')
        plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    
    # New filename format
    if dice_score is not None:
        filename = f'dice{dice_score:.3f}_sample{index:04d}.png'
    else:
        filename = f'sample{index:04d}.png'
    
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=150)
    plt.close()
    

def run_inference(model, test_loader, device, output_dir, metric_fn):
    """Run inference and evaluate results."""
    dice_scores = []
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_loader, desc="Processing")):
            # Get data
            x = sample["x"]
            y_true = sample["y_true"]
            
            # Move to device
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            
            # Forward pass
            y_pred = model(x) # logits
            dice_score = metric_fn(y_pred, y_true.bool())
            dice_scores.append(dice_score.item())
            
            y_pred = torch.sigmoid(y_pred) # prob 0-1
            
            # Get binary prediction
            # pred_mask = (y_pred > 0.5)
            
            # Save visualization
            save_prediction(
                x.cpu().numpy(),               # Shape: [1, 1, H, W]
                y_true.cpu().numpy(),          # Shape: [1, 1, H, W]
                y_pred.cpu().numpy(),          # Shape: [1, 1, H, W]
                output_dir,
                idx,
                dice_score.item()
            )
    
    # Save dice scores
    scores_file = os.path.join(output_dir, 'dice_scores.csv')
    with open(scores_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Dice Score'])
        for idx, score in enumerate(dice_scores):
            writer.writerow([f'Image_{idx:04d}', f'{score:.4f}'])
    
    # Print summary statistics
    print(f"\nDice Scores Summary:")
    print(f"Mean: {np.mean(dice_scores):.4f}")
    print(f"Std: {np.std(dice_scores):.4f}")
    print(f"Min: {np.min(dice_scores):.4f}")
    print(f"Max: {np.max(dice_scores):.4f}")
    print(f"\nDetailed scores saved to: {scores_file}")
    
    return dice_scores

def main():
    parser = argparse.ArgumentParser(description='Run inference on test data')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    args = parser.parse_args()
    
    # Load configuration
    config_args, path_exp_base, exp_name = load_config(args.config, os.path.dirname(args.config))
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.model, config_args["n_features"], device)
    
    # Create test data loader
    test_loader = load_test_data(config_args)
    
    # Create metric function
    metric_fn = CustomDiceCoefficientWithLogits() #DiceCoefficientWithLogits()
    
    # Run inference and evaluation
    dice_scores = run_inference(model, test_loader, device, args.output, metric_fn)

if __name__ == '__main__':
    main()