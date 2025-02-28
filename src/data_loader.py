# data_loader.py
import h5py
import numpy as np
import torch.utils.data as utils
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from unet2d.data import AugTransformConstantPosWeights, AugTransformCenterToEdgeWeights, Dataset

def create_augmentation(augmentation_params):
    """Create augmentation pipeline from augmentation_params dict."""

    # Graceful fallback if augmentation_params is missing or empty
    if not augmentation_params:
        print("No augmentation parameters found; returning identity augmentation.")
        return iaa.Noop()

    aug_train = iaa.Sequential([
        iaa.TranslateX(px=tuple(augmentation_params["TranslateX"]["px"])),
        iaa.TranslateY(px=tuple(augmentation_params["TranslateY"]["px"])),
        iaa.Fliplr(augmentation_params["Fliplr"]),
        iaa.Flipud(augmentation_params["Flipud"]),
        iaa.GaussianBlur(sigma=tuple(augmentation_params["GaussianBlur"]["sigma"])),
        iaa.LinearContrast(tuple(augmentation_params["LinearContrast"])),
        iaa.Multiply(tuple(augmentation_params["Multiply"])),
        iaa.Sometimes(
            augmentation_params["Sometimes"],
            iaa.ElasticTransformation(
                alpha=tuple(augmentation_params["ElasticTransformation"]["alpha"]),
                sigma=augmentation_params["ElasticTransformation"]["sigma"]
            )
        ),
        iaa.Affine(
            scale=iap.Normal(
                augmentation_params["Affine"]["scale_mean"],
                augmentation_params["Affine"]["scale_std"]
            ),
            rotate=tuple(augmentation_params["Affine"]["rotate"]),
            shear=iap.Normal(
                augmentation_params["Affine"]["shear_mean"],
                augmentation_params["Affine"]["shear_std"]
            )
        )
    ])
    return aug_train


def min_max_by_percentile(img_data, min_percentile=50, max_percentile=99):
    """
    Apply min_max scaling to img_data such that 0 = p50, 1 = p99 by default
    Anything outside the range of [0,1] is clipped
    """
    img_data_min = np.percentile(img_data, min_percentile)
    img_data_max = np.percentile(img_data, max_percentile)
    img_data_normalized = (img_data - img_data_min) / (img_data_max - img_data_min)
    img_data_normalized = np.clip(img_data_normalized, 0, 1)
    return img_data_normalized


    
    load_train_val_data(
        config_args,
        os.path.join(path_exp_base, "dataloaders") # savepath for augmented data; if None, nothing is saved
    )
    

def load_train_val_data(config_args, visualization_dir=None):   
    """Load training and validation data according to what is specified in config.yaml."""
    
    list_path_h5 = config_args["list_path_h5"]
    frac_val = config_args["frac_val"]
    frac_test = config_args["frac_test"]
    batch_size = config_args["batch_size"]
    loss_weight = config_args["loss_weight"]
    augmentation_parameters = config_args["augmentation"]
        
    list_img = []
    list_label = []
    idx_val = []
    idx_train = []
    curr_idx = 0  # Track the current index in our final lists
    
    for path_h5 in list_path_h5:
        with h5py.File(path_h5, "r") as h5f:
            dataset_names = list(h5f.keys())
            
            # calculate number of test frames and validation frames
            n_total = len(dataset_names)
            n_test = int(np.floor(n_total * frac_test))
            n_val = int(np.floor(n_total * frac_val))
            
            for i, datasetname_ in enumerate(dataset_names):
                # Skip test samples completely
                if i < n_test:
                    continue
                
                # Load the data
                dataset_ = h5f[datasetname_]
                img_data = (dataset_["img"][()]).astype(np.float32)
                img_data = min_max_by_percentile(img_data) # normalize images
                img = np.moveaxis(np.expand_dims(img_data, axis=2), [0,1,2], [2,1,0])
                list_img.append(img)
                
                label_data = dataset_["label"][()].astype(bool)
                label = np.moveaxis(np.expand_dims(label_data, axis=2), [0,1,2], [2,1,0])
                list_label.append(label)
                
                # Assign to validation or training set
                if i < n_test + n_val: # Validation samples
                    idx_val.append(curr_idx)
                else: # Training samples
                    idx_train.append(curr_idx)
                
                curr_idx += 1  # Increment our index counter
    
    # Convert to numpy arrays and check for disjoint sets
    idx_val = np.array(idx_val)
    idx_train = np.array(idx_train)
    assert len(np.intersect1d(idx_train, idx_val)) == 0, "Train-val split have repeated frames!"
    print(f"{len(idx_train)} frames go to train set. {len(idx_val)} frames go to validation set.")
    
    X_train = np.concatenate(list(map(list_img.__getitem__, idx_train)), axis=0)
    Y_train = np.concatenate(list(map(list_label.__getitem__, idx_train)), axis=0)
    X_val = np.concatenate(list(map(list_img.__getitem__, idx_val)), axis=0)
    Y_val = np.concatenate(list(map(list_label.__getitem__, idx_val)), axis=0)
    
    # Create augmentations for training set
    aug_train = create_augmentation(augmentation_parameters)
    # tfm_train = AugTransformConstantPosWeights(aug_train, loss_weight)
    # tfm_val = AugTransformConstantPosWeights(iaa.Identity(), loss_weight)

    # Apply center to edge weights for pixel-wise loss function
    center_weight = 5.0
    tfm_train = AugTransformCenterToEdgeWeights(aug_train, center_weight)
    tfm_val = AugTransformCenterToEdgeWeights(iaa.Identity(), center_weight)
    
    # Create datasets
    data_train = Dataset(X_train, Y_train, tfm_train)
    data_val = Dataset(X_val, Y_val, tfm_val)
    
    # Create loaders
    train_loader = utils.DataLoader(data_train, shuffle=True, num_workers=0, batch_size=batch_size)
    val_loader = utils.DataLoader(data_val, shuffle=False, num_workers=0, batch_size=batch_size)

    # Visualize datasets if requested
    if visualization_dir is not None:
        visualize_dataset(train_loader, 'train', visualization_dir)
        visualize_dataset(val_loader, 'val', visualization_dir)
    
    return {"train": train_loader, "val": val_loader}

    
def visualize_dataset(data_loader, dataset_name, output_dir):
    """Visualize all samples in a dataset and save to output_dir."""
    import matplotlib.pyplot as plt
    import os
    
    # Create output directories
    dataset_dir = os.path.join(output_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, 'images')
    weight_dir = os.path.join(dataset_dir, 'weights')
    overlay_dir = os.path.join(dataset_dir, 'overlays')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    print(f"Visualizing {dataset_name} dataset...")
    sample_idx = 0
    
    # Process each batch
    for batch_idx, batch in enumerate(data_loader):
        # Get data from batch
        images = batch['x'].numpy()
        labels = batch['y_true'].numpy()
        weights = batch['weights'].numpy()
        
        # Process each sample in the batch
        for i in range(images.shape[0]):
            # Extract individual components
            img = images[i, 0]  # First channel
            label = labels[i, 0]
            weight = weights[i, 0]
            
            # Save image
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, f'sample_{sample_idx:04d}_image.png'), 
                        bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            
            # Save weight map with colorbar
            plt.figure(figsize=(7, 6))
            im = plt.imshow(weight, cmap='viridis')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f'Weight Map (Max: {weight.max():.2f})')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(weight_dir, f'sample_{sample_idx:04d}_weight.png'), 
                        bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            
            # Save overlay of label on image
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.imshow(label, cmap='Reds', alpha=0.5)  # Red overlay for labels
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(overlay_dir, f'sample_{sample_idx:04d}_overlay.png'), 
                        bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            sample_idx += 1
    print(f"Saved {sample_idx} visualizations for {dataset_name} dataset to {output_dir}")


def load_test_data(config_args, batch_size=1):
    """Load test data only."""
    list_path_h5 = config_args["list_path_h5"]
    frac_test = config_args["frac_test"]
    loss_weight = config_args["loss_weight"]
    
    list_img = []
    list_label = []
    for path_h5 in list_path_h5:
        with h5py.File(path_h5, "r") as h5f:
            dataset_names = list(h5f.keys())
            n_total = len(dataset_names)
            n_test = np.floor(n_total * frac_test)
            for i, datasetname_ in enumerate(dataset_names):
                if i >= n_test:  # Only load first n_test samples
                    break
                dataset_ = h5f[datasetname_]
                img_data = (dataset_["img"][()]).astype(np.float32)
                img_data = min_max_by_percentile(img_data)
                img = np.moveaxis(np.expand_dims(img_data, axis=2), [0,1,2], [2,1,0])
                list_img.append(img)
                
                label_data = dataset_["label"][()].astype(bool)
                label = np.moveaxis(np.expand_dims(label_data, axis=2), [0,1,2], [2,1,0])
                list_label.append(label)

    print(f"{len(list_img)} frames go to test set.")
    X_test = np.concatenate(list_img, axis=0)
    Y_test = np.concatenate(list_label, axis=0)
    
    # Create test dataset with identity transform
    tfm_test = AugTransformConstantPosWeights(iaa.Identity(), loss_weight)
    data_test = Dataset(X_test, Y_test, tfm_test)
    
    # Create test loader
    test_loader = utils.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return test_loader