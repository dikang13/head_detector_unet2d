import torch
import torch.nn as nn

def dice_coefficient(y_pred, y_true, epsilon=1e-6):
    y_pred = y_pred.flatten(start_dim=1)
    y_true = y_true.flatten(start_dim=1)
        
    intersaction = (2 * y_pred * y_true)
    cardinality = (y_pred + y_true + epsilon)
        
    result = intersaction.sum(dim=1) / cardinality.sum(dim=1)

    return result

def dice_coefficient_multi(y_pred, y_true, n_class, epsilon=1e-6):
    y_pred = nn.functional.one_hot(y_pred, n_class)
    y_true = nn.functional.one_hot(y_true, n_class)

    return dice_coefficient(y_pred, y_true, epsilon)

def dice_coefficient_with_logits(y, y_true, epsilon=1e-6):
    y_pred = torch.sigmoid(y)
    
    return dice_coefficient(y_pred, y_true, epsilon)

def dice_coefficient_with_logits_multi(y, y_true, n_class, epsilon=1e-6):
    y_pred = torch.argmax(torch.nn.functional.softmax(y, dim=1), 1)
    
    return dice_coefficient_multi(y_pred, y_true, n_class, epsilon)

class DiceCoefficient(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceCoefficient, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        dice = dice_coefficient(y_pred, y_true, self.epsilon)
        
        return dice.mean()

class DiceCoefficientMulti(nn.Module):
    def __init__(self, n_class, epsilon=1e-6):
        super(DiceCoefficientMulti, self).__init__()
        self.epsilon = epsilon
        self.n_class = n_class
        
    def forward(self, y_pred, y_true):
        dice = dice_coefficient_multi(y_pred, y_true,
                                      self.n_class, self.epsilon)
        
        return dice.mean()

class DiceCoefficientWithLogits(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceCoefficientWithLogits, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y, y_true):
        dice = dice_coefficient_with_logits(y, y_true, self.epsilon)
        
        return dice.mean()
        
class CustomDiceCoefficientWithLogits(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CustomDiceCoefficientWithLogits, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y, y_true, weights=None):
        # Get per-sample dice scores first (same as original)
        dice = dice_coefficient_with_logits(y, y_true, self.epsilon)
        
        # Apply the transformation to each dice score
        transformed_dice = torch.zeros_like(dice)
        
        # Create a mask for values below threshold
        low_mask = dice < 0.05
        
        # Apply transformations based on masks
        transformed_dice[low_mask] = 0.0 # heavily penalize misses or near-misses
        transformed_dice[~low_mask] = torch.sqrt(dice[~low_mask]) # compress successes to a narrower range
        
        # Return mean of transformed values (same format as original)
        return transformed_dice.mean()

class DiceCoefficientWithLogitsMulti(nn.Module):
    def __init__(self, n_class, epsilon=1e-6):
        super(DiceCoefficientWithLogitsMulti, self).__init__()
        self.epsilon = epsilon
        self.n_class = n_class
        
    def forward(self, y, y_true):
        dice = dice_coefficient_with_logits_multi(y, y_true,
                                                  self.n_class, self.epsilon)
        
        return dice.mean()
