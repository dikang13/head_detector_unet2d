# train.py
import argparse
import torch
import os
from config import load_config
from data_loader import load_train_val_data
from model import setup_model
from unet2d.train import train_model

def main():
    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to H5 data file')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    
    # Load configuration
    config_args, path_exp_base, exp_name = load_config(args.config, args.output)
    
    # Load data
    data_loader = load_train_val_data(
        args.data, 
        config_args["n_test"], 
        config_args["n_val"], 
        config_args["scaling_factor"],
        config_args["batch_size"],
        config_args["loss_weight"],
        config_args["augmentation"]
    )
    
    # Setup model
    model, optimizer, scheduler, loss_fn, metric_fn = setup_model(config_args, device)
    
    # Train model
    best_weights = train_model(
        model, device, data_loader,
        loss_fn, metric_fn,
        optimizer, scheduler,
        config_args, True, False
    )
    
    # Save models
    savepath_best = os.path.join(path_exp_base, exp_name + "_best.pt")
    savepath_final = os.path.join(path_exp_base, exp_name + "_final.pt")
    
    torch.save(best_weights, savepath_best)
    torch.save(model.state_dict(), savepath_final)
    print(f"Best weights saved to: {savepath_best}")
    print(f"Final weights saved to: {savepath_final}")

if __name__ == "__main__":
    main()