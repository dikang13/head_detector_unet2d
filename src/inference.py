# inference.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet2d.model.unet_model import UNet2D
from unet2d.model.metrics import DiceCoefficientWithLogits
from data_loader import load_test_data
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

def compute_dice_score(pred, target):
    """
    Compute Dice score between binary masks.
    pred and target should be binary tensors.
    """
    smooth = 1e-5
    intersection = (pred & target).float().sum()
    union = pred.float().sum() + target.float().sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

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
            y_pred = torch.sigmoid(y_pred) # prob 0-1
            
            # Get binary prediction
            pred_mask = (y_pred > 0.5)
            
            # Calculate binary dice score
            dice_score = compute_dice_score(pred_mask, y_true.bool())
            dice_scores.append(dice_score.item())
            
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
    parser.add_argument('--data', type=str, required=True,
                       help='Path to H5 data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--n-test', type=int, default=500,
                       help='Number of test samples')
    parser.add_argument('--n-features', type=int, default=64,
                       help='Number of features in UNet')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, args.n_features, device)
    
    # Create test data loader
    test_loader = load_test_data(
        args.data, 
        args.n_test, 
        args.scaling_factor,
        batch_size=1  # Use batch size 1 for inference
    )
    
    # Create metric function
    metric_fn = DiceCoefficientWithLogits()
    
    # Run inference and evaluation
    dice_scores = run_inference(model, test_loader, device, args.output, metric_fn)

if __name__ == '__main__':
    main()