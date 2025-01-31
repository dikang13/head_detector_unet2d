# model.py
import torch
import torch.optim as optim
from unet2d.model.unet_model import UNet2D
from unet2d.model.losses import PixelwiseBCEWithLogitsLoss
from unet2d.model.metrics import DiceCoefficientWithLogits
import os

def setup_model(args, device):
    """Initialize model, optimizer, and loss functions."""
    model = UNet2D(1, 1, args["n_features"], False)
    model.to(device)

    print("Trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))    
    
    model_optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer,
        patience=args["lr_scheduler_patience"],
        factor=args["lr_scheduler_factor"]
    )
    
    f_loss = PixelwiseBCEWithLogitsLoss()
    f_metric = DiceCoefficientWithLogits()
    
    if os.path.isfile(args["path_checkpoint"]):
        print("Continuing training from checkpoint")
        dict_checkpoint = torch.load(args["path_checkpoint"])
        args["start_epoch"] = dict_checkpoint["epoch"] + 1
        model.load_state_dict(dict_checkpoint["model_state_dict"])
        model_optimizer.load_state_dict(dict_checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(dict_checkpoint["scheduler_state_dict"])
    
    return model, model_optimizer, lr_scheduler, f_loss, f_metric