import torch
import numpy as np

def compute_miou(pred, target, num_classes=21, ignore_index=255):
    """
    Compute mean Intersection over Union (mIoU)
    Args:
        pred: (batch_size, H, W) tensor of predicted class IDs
        target: (batch_size, H, W) tensor of ground truth class IDs
        num_classes: number of classes
        ignore_index: index to ignore in target
    Returns:
        miou: mean IoU over all classes
    """
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Ignore specified index
    mask = target_flat != ignore_index
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    # Compute IoU for each class
    iou = []
    for cls in range(num_classes):
        pred_cls = pred_flat == cls
        target_cls = target_flat == cls
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union > 0:
            iou.append(intersection / union)
    
    # Compute mean IoU
    if iou:
        return np.mean(iou)
    else:
        return 0.0

def compute_pixel_accuracy(pred, target, ignore_index=255):
    """
    Compute pixel accuracy
    Args:
        pred: (batch_size, H, W) tensor of predicted class IDs
        target: (batch_size, H, W) tensor of ground truth class IDs
        ignore_index: index to ignore in target
    Returns:
        accuracy: pixel accuracy
    """
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Ignore specified index
    mask = target_flat != ignore_index
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    # Compute accuracy
    correct = (pred_flat == target_flat).sum().item()
    total = len(pred_flat)
    
    if total > 0:
        return correct / total
    else:
        return 0.0