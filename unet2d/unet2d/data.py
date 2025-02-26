import numpy as np
import imgaug as ia
import torch.utils as utils

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

class AugTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, y_true):
        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # normalize
        x_aug = normalize(x_aug)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.float32)

        return {"x": x_aug, "y_true": y_true_aug}

class AugTransformConstantPosWeights:
    """creates weights with foreground to 1 * pos_weight and background to 1"""
    def __init__(self, aug, pos_weight):
        self.aug = aug
        self.pos_weight = pos_weight

    def __call__(self, x, y_true):
        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # normalize
        x_aug = normalize(x_aug)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.float32)
        weights = self.pos_weight * (y_true_aug > 0).astype(np.float32) + 1

        return {"x": x_aug, "y_true": y_true_aug, "weights": weights}

class AugTransformCenterToEdgeWeights:
    """Creates weights that are highest at object centers and taper to 1.0 at boundaries"""
    def __init__(self, aug, center_weight=5.0):
        self.aug = aug
        self.center_weight = center_weight
        
    def __call__(self, x, y_true):
        # Augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)
        
        # Get binary mask
        binary_mask = y_true_aug.arr[:, :, 0].astype(np.uint8)
        
        # Create weights (start with all ones - background weight)
        weights = np.ones_like(binary_mask, dtype=np.float32)
        
        # If there's a labeled region
        if np.any(binary_mask > 0):
            # Calculate distance transform from background
            from scipy.ndimage import distance_transform_edt
            dist_from_bg = distance_transform_edt(binary_mask)
            
            # Normalize distances within each object to 0-1 range
            # This creates a weight that's 1.0 at boundaries and increases toward center
            if np.max(dist_from_bg) > 0:
                normalized_dist = dist_from_bg / np.max(dist_from_bg)
                
                # Scale to desired weight range: 1.0 at edges to center_weight at centers
                scaled_weights = 1.0 + (self.center_weight - 1.0) * normalized_dist
                
                # Apply only within foreground regions
                weights[binary_mask > 0] = scaled_weights[binary_mask > 0]
        
        # Normalize image
        x_aug = normalize(x_aug)
        
        # Reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(binary_mask, 0).astype(np.float32)
        weights = np.expand_dims(weights, 0).astype(np.float32)
        
        return {"x": x_aug, "y_true": y_true_aug, "weights": weights}


class AugTransformMultiClass:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, y_true):
        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # normalize
        x_aug = normalize(x_aug)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.long)
        weights = (y_true_aug >= 0).astype(np.float32)

        return {"x": x_aug, "y_true": y_true_aug, "weights": weights}

class Dataset(utils.data.Dataset):
    def __init__(self, x_array, y_true_array, transform):
        self.x_array = x_array
        self.y_true_array = y_true_array
        self.data_len = x_array.shape[0]
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        x = self.x_array[idx, :, :]
        y_true = self.y_true_array[idx, :, :]
        sample = self.transform(x, y_true)

        return sample
