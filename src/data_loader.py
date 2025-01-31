# data_loader.py
import h5py
import numpy as np
import torch.utils.data as utils
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from unet2d.data import AugTransformConstantPosWeights, Dataset


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


def load_train_val_data(path_h5, n_test, n_val, scaling_factor, batch_size=64, loss_weight=1.0, augmentation_parameters=None):
    """Load training and validation data."""
    with h5py.File(path_h5, "r") as h5f:
        dataset_names = list(h5f.keys())
        
    list_img = []
    list_label = []
    with h5py.File(path_h5, "r") as h5f:
        for datasetname_ in dataset_names:
            dataset_ = h5f[datasetname_]
            img_data = (dataset_["img"][()]/scaling_factor).astype(np.float32)
            img = np.moveaxis(np.expand_dims(img_data, axis=2), [0,1,2], [2,1,0])
            list_img.append(img)
            
            label_data = dataset_["label"][()].astype(bool)
            label = np.moveaxis(np.expand_dims(label_data, axis=2), [0,1,2], [2,1,0])
            list_label.append(label)
    
    # Only get validation and training indices
    idx_val = np.arange(n_test, n_test+n_val)
    idx_train = np.arange(n_test+n_val, len(list_img))
    
    X_val = np.concatenate(list(map(list_img.__getitem__, idx_val)), axis=0)
    Y_val = np.concatenate(list(map(list_label.__getitem__, idx_val)), axis=0)
    X_train = np.concatenate(list(map(list_img.__getitem__, idx_train)), axis=0)
    Y_train = np.concatenate(list(map(list_label.__getitem__, idx_train)), axis=0)
    
    print(f"Training data type: {X_train.dtype}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training label shape: {Y_train.shape}")
    
    # Create augmentations
    aug_train = create_augmentation(augmentation_parameters)
    tfm_train = AugTransformConstantPosWeights(aug_train, loss_weight)
    tfm_val = AugTransformConstantPosWeights(iaa.Identity(), loss_weight)
    
    # Create datasets
    data_train = Dataset(X_train, Y_train, tfm_train)
    data_val = Dataset(X_val, Y_val, tfm_val)
    
    # Create loaders
    train_loader = utils.DataLoader(data_train, shuffle=True, num_workers=0, batch_size=batch_size)
    val_loader = utils.DataLoader(data_val, shuffle=False, num_workers=0, batch_size=batch_size)
    
    return {"train": train_loader, "val": val_loader}


def load_test_data(path_h5, n_test, scaling_factor, batch_size=1, loss_weight=1.0):
    """Load test data only."""
    with h5py.File(path_h5, "r") as h5f:
        dataset_names = list(h5f.keys())
        
    list_img = []
    list_label = []
    with h5py.File(path_h5, "r") as h5f:
        for i, datasetname_ in enumerate(dataset_names):
            if i >= n_test:  # Only load first n_test samples
                break
            dataset_ = h5f[datasetname_]
            img_data = (dataset_["img"][()]/scaling_factor).astype(np.float32)
            img = np.moveaxis(np.expand_dims(img_data, axis=2), [0,1,2], [2,1,0])
            list_img.append(img)
            
            label_data = dataset_["label"][()].astype(bool)
            label = np.moveaxis(np.expand_dims(label_data, axis=2), [0,1,2], [2,1,0])
            list_label.append(label)
    
    X_test = np.concatenate(list_img, axis=0)
    Y_test = np.concatenate(list_label, axis=0)
    
    print(f"Test data type: {X_test.dtype}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test label shape: {Y_test.shape}")
    
    # Create test dataset with identity transform
    tfm_test = AugTransformConstantPosWeights(iaa.Identity(), loss_weight)
    data_test = Dataset(X_test, Y_test, tfm_test)
    
    # Create test loader
    test_loader = utils.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return test_loader