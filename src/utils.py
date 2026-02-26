import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Callable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import complete_box_iou
import torchvision.transforms as transforms

DEFAULT_SEED = 101

MEAN=(0.485, 0.456, 0.406)
STD=(0.229, 0.224, 0.225)

def get_device():
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"

def seed_everything(seed: int = DEFAULT_SEED):
    # Python RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # Torch RNG (CPU)
    torch.manual_seed(seed)

    # Torch RNG (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUDA matmul reproducibility (PyTorch 2+)
    # torch.use_deterministic_algorithms(True)

    # Hash seed (important for dataloader shuffling!)
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders_for(train_dataset: Dataset, test_dataset: Dataset, batch_size: int, collate_fn: Callable | None = None, seed: int = DEFAULT_SEED):
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
        )
    
    if test_dataset is not None:
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_fn
            )
            
        return train_dataloader, test_dataloader
    
    return train_dataloader

def to_binary_masks(values):
    sm_values = torch.softmax(values, dim=1)
    max_indices = torch.argmax(sm_values, dim=1, keepdim=True)

    # Create a binary mask with 1 at max indices and 0 elsewhere
    binary_masks = torch.zeros_like(sm_values).scatter_(1, max_indices, 1)

    return binary_masks

def masks_to_targets(masks_batch, device):
    """
    Converts a batch of padded instance masks (B, N_ATT, H, W) into 
    a List[Dict] format required by the RetinaNet detection head.
    
    Bounding boxes are calculated from non-zero masks and returned in 
    the standard [x1, y1, x2, y2] format (absolute coordinates).
    Labels are fixed at 1 for all detected instances.
    """
    targets_list = []
    
    for i in range(masks_batch.shape[0]): # Loop through the batch
        masks = masks_batch[i] # (N_ATT, H, W)
        
        boxes = []
        labels = []
        
        # Iterate over up to N_ATT padded masks
        for j in range(masks.shape[0]):
            mask = masks[j]
            # Check if this mask is a valid instance (not a zero-padding mask)
            if mask.sum() > 1e-6:
                # Find coordinates of non-zero pixels
                # NOTE: torch.where returns two tuples: (y_coords, x_coords)
                ys, xs = torch.where(mask)
                
                # Bounding box in [x1, y1, x2, y2] format
                x_min = xs.min().item()
                y_min = ys.min().item()
                x_max = xs.max().item()
                y_max = ys.max().item()
                
                boxes.append([x_min, y_min, x_max, y_max])
                # Label is always 1 for the first object class (crater)
                labels.append(1)
        
        # Assemble tensors for the RetinaNet target dictionary
        if not boxes:
            # If no instances are present, create an empty tensor pair
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels_tensor = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
            labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)

        targets_list.append({
            'boxes': boxes_tensor,
            'labels': labels_tensor
        })
        
    return targets_list

def create_gt_panoptic_map(mod_masks_batch, n_att, n_stuff):
    """
    Constructs the Panoptic Ground Truth map (GT_pan) from the fixed-size 
    list of ground-truth masks (mod_masks), where each mask corresponds to 
    an output channel index.
    
    The mod_masks for one image should have shape (n_out, H, W).
    """
    
    gt_panoptic_maps = []
    
    # Total number of channels / masks to process
    n_out = n_att + n_stuff + 2 
    
    for mod_masks in mod_masks_batch:
        # mod_masks has shape (n_out, H, W). The masks are mutually exclusive.
        _, H, W = mod_masks.shape
        device = mod_masks.device
        
        # 1. Initialize GT map with the Unlabeled/Background index (the very last channel index)
        background_idx = n_out - 1
        gt_pan = torch.full((H, W), background_idx, dtype=torch.long, device=device)
        
        # 2. Iterate through all mask channels (excluding the last one which is the background itself)
        # and assign the corresponding channel index k to the map where the mask is active.
        for k in range(n_out - 1):
            mask = mod_masks[k].bool()
            gt_pan[mask] = k
            
        gt_panoptic_maps.append(gt_pan)

    return torch.stack(gt_panoptic_maps, dim=0)

def convert_preds_to_pq_format(upscaled_output, n_att, crater_class_id):
    device = upscaled_output.device
    B, _, H, W = upscaled_output.shape

    pred_index_map = torch.argmax(upscaled_output, dim=1)  # (B, H, W)
    pq_preds = torch.zeros((B, H, W, 2), dtype=torch.long, device=device)

    for k in range(n_att):
        mask_k = (pred_index_map == k)  # (B, H, W)
        # Reshape mask_k to (B, H, W, 1) to broadcast on last dim of pq_preds
        mask_k_expanded = mask_k.unsqueeze(-1).expand(-1, -1, -1, 2)
        # Assign category and instance ID across channels using tensor masking and indexing
        pq_preds[mask_k_expanded] = torch.tensor([crater_class_id, k + 1], device=device).repeat(mask_k.sum().item(), 1).view(-1)

    k_unmatched = n_att
    mask_unmatched = (pred_index_map == k_unmatched)
    mask_unmatched_expanded = mask_unmatched.unsqueeze(-1).expand(-1, -1, -1, 2)
    pq_preds[mask_unmatched_expanded] = torch.tensor([crater_class_id, n_att + 1], device=device).repeat(mask_unmatched.sum().item(), 1).view(-1)

    return pq_preds

def convert_gt_to_pq_format(gt_pan_map, n_att, crater_class_id):
    """
    Converts the panoptic ground truth index map (B, H, W) to PQ format (B, H, W, 2).
    """
    device = gt_pan_map.device
    B, H, W = gt_pan_map.shape

    # Initialize (Background Category, Instance 0)
    pq_target = torch.zeros((B, H, W, 2), dtype=torch.long, device=device)

    # Mask for foreground (not background)
    mask_foreground = (gt_pan_map != (n_att + 1))

    # Assign category ID for foreground pixels (channel 0)
    pq_target[..., 0][mask_foreground] = crater_class_id
    
    # Assign instance IDs for foreground pixels (channel 1)
    instance_indices = gt_pan_map[mask_foreground]
    pq_target[..., 1][mask_foreground] = instance_indices + 1

    return pq_target

def mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
    # output in [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
    rowmap = (mask.sum(dim=0) != 0).int()
    colmap = (mask.sum(dim=1) != 0).int()
    x1_scaled = rowmap.argmax()
    y1_scaled = colmap.argmax()
    x2_scaled = rowmap.shape[0] - rowmap.flip(0).argmax() - 1
    y2_scaled = colmap.shape[0] - colmap.flip(0).argmax() - 1
    return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]

def masks_list_to_targets_list(masks_list: torch.Tensor) -> list:
    targets_list = []
                
    for masks in masks_list:
        boxes = torch.tensor([mask_to_bbox(mask) for mask in masks if mask.sum() != 0]).float()
        labels = torch.ones(boxes.shape[0], dtype=torch.int64)

        targets = {
            "boxes": boxes,
            "labels": labels
        }
        targets_list.append(targets)

    return targets_list

def collate_fn(batch):
    # 'batch' is a list of tuples: [(image1, target1), (image2, target2), ...]
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack images into a single tensor (B, C, H, W)
    images = torch.stack(images, 0)
    
    # Targets remain a list of dictionaries (List[Dict])
    return images, targets

def visualize_detections_masks(image, masks, denormalize=True):
    denormalize_fn = transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)], 
        std=[1/s for s in STD]
    )

    if denormalize:
        image = denormalize_fn(image)

    image = image.cpu().numpy().transpose(1, 2, 0)

    plt.imshow(image)
    n = masks.shape[0]
    colors = plt.get_cmap('hsv', n)
    for i in range(n):
        mask = masks[i].numpy()
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
        colored_mask[:, :, :3] = colors(i)[:3]  # RGB
        colored_mask[:, :, 3] = mask * 0.4      # Alpha
        plt.imshow(colored_mask, interpolation='none')


def visualize_detections_targets(image_tensor, targets_dict, show_labels=True, denormalize=True):
    """
    Denormalizes an image tensor and displays it with ground truth bounding boxes.

    Args:
        image_tensor (torch.Tensor): Normalized image tensor (C, H, W).
        targets_dict (dict): Dictionary containing 'boxes' (N, 4) in [x1, y1, x2, y2]
                             and 'labels' (N,).
        mean (tuple): Mean values used for normalization.
        std (tuple): Standard deviation values used for normalization.
        class_name (str): Name of the detected class (Crater).
    """
    
    # 1. Denormalize and Convert Tensor to NumPy Array (H, W, C)
    
    # Reverse Normalization: value = (normalized_value * std) + mean
    # We use a custom lambda transform to ensure the process is correctly applied channel-wise
    denormalize_fn = transforms.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)], 
        std=[1/s for s in STD]
    )

    if denormalize:
        image_tensor = denormalize_fn(image_tensor)

    img_array = image_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Clamp values to the valid [1] range in case of floating point inaccuracies
    img_array = np.clip(img_array, 0, 1)

    # 2. Extract Bounding Boxes
    # RetinaNet targets use absolute pixel coordinates [x1, y1, x2, y2] [2].
    boxes = targets_dict['boxes'].cpu().numpy()
    labels = targets_dict['labels'].cpu().numpy()
    
    # Get image dimensions (after resizing applied in the dataset)
    H, W, C = img_array.shape

    # 3. Plotting
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_array)
    
    for box, label in zip(boxes, labels):
        # Coordinates are [x1, y1, x2, y2]
        x_min, y_min, x_max, y_max = box
        
        # Calculate width and height for Matplotlib patch
        width = x_max - x_min
        height = y_max - y_min
        
        # Create a Rectangle patch (start point is (x_min, y_min))
        rect = patches.Rectangle(
            (x_min, y_min), 
            width, 
            height,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label text (Crater, since category_id=1 [3] corresponds to the first class)
        if show_labels:
            ax.text(
                x_min, y_min - 5, # Position slightly above the box
                label,
                color='red',
                fontsize=8,
                bbox=dict(facecolor='none', alpha=0.5, edgecolor='none')
            )

    ax.set_title(f"Image Visualization ({len(boxes)} detections)")
    plt.show()

def remove_duplicates(targets, iou_threshold=-0.1):
    """complete_box_iou
    Remove duplicate bounding boxes based on IoU threshold.

    Args:
        targets (dict): Dictionary containing 'boxes' (N, 4) and 'labels' (N,).
        iou_threshold (float): IoU threshold to consider boxes as duplicates.

    Returns:
        dict: Dictionary with unique 'boxes' and 'labels'.
    """

    boxes = targets["boxes"]
    labels = targets["labels"]
    scores = targets["scores"]

    # sort boxes by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    labels = labels[sorted_indices]
    scores = scores[sorted_indices]

    if len(boxes) == 0:
        return targets

    # Compute IoU between all pairs of boxes
    iou_matrix = complete_box_iou(boxes, boxes)

    # Create a mask to keep track of which boxes to keep
    keep_mask = torch.ones(len(boxes), dtype=torch.bool)

    # remove iou duplicates
    for i in range(len(boxes)):
        if not keep_mask[i]:
            continue
        # Mark boxes that have high IoU with the current box as duplicates
        for j in range(len(boxes)):
            if i != j and iou_matrix[i, j] > iou_threshold:
                keep_mask[j] = False

    # remove boxes that cover other boxes (if box A covers box B, we keep the one with the higher score)
    for i in range(len(boxes)):
        if not keep_mask[i]:
            continue
        for j in range(len(boxes)):
            if i != j and keep_mask[j]:
                # Check if box i covers box j
                if (boxes[i, 0] <= boxes[j, 0] and boxes[i, 1] <= boxes[j, 1] and
                    boxes[i, 2] >= boxes[j, 2] and boxes[i, 3] >= boxes[j, 3]):
                    # If box i has a higher score than box j, we keep box i and remove box j
                    if scores[i] >= scores[j]:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
                        break

    # Filter out duplicates
    unique_boxes = boxes[keep_mask]
    # new_labels = torch.tensor(list(range(len(unique_boxes))))
    new_labels = labels[keep_mask]
    unique_scores = scores[keep_mask]

    return {'boxes': unique_boxes, 'labels': new_labels, 'scores': unique_scores}

def remove_low_confidence_edge_boxes(targets, height, width, confidence_threshold=0.5, edge_threshold_factor=0.05):
    """
    Remove bounding boxes that are near the edges of the image and have low confidence scores.

    Args:
        targets (dict): Dictionary containing 'boxes' (N, 4) and 'scores' (N,).
        height (int): Height of the image.
        width (int): Width of the image.
        confidence_threshold (float): Confidence score threshold below which boxes will be removed.
        edge_threshold_factor (float): Factor of image dimensions to consider as edge threshold.

    Returns:
        dict: Dictionary with filtered 'boxes', 'labels', and 'scores'.
    """
    boxes = targets["boxes"]
    labels = targets["labels"]
    scores = targets["scores"]

    if len(boxes) == 0:
        return targets

    keep_mask = torch.ones(len(boxes), dtype=torch.bool)

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i]
        score = scores[i]

        # Check if the entirety of the box is within edge_threshold from the image borders
        near_edge = (x_min < edge_threshold_factor * width or 
                     y_min < edge_threshold_factor * height or
                     x_max > (1 - edge_threshold_factor) * width or
                     y_max > (1 - edge_threshold_factor) * height)

        # If the box is near the edge and has a low confidence score, mark it for removal
        if near_edge and score < confidence_threshold:
            keep_mask[i] = False

    filtered_boxes = boxes[keep_mask]
    filtered_labels = labels[keep_mask]
    filtered_scores = scores[keep_mask]

    return {'boxes': filtered_boxes, 'labels': filtered_labels, 'scores': filtered_scores}

def inflate_annots(targets, height, width, height_factor, width_factor):
    """
    Inflates bounding boxes by a specified factor.

    Args:
        targets (dict): Dictionary containing 'boxes' (N, 4) and 'labels' (N,).
        height (int): Height of the image.
        width (int): Width of the image.
        height_factor (float): Factor by which to inflate each side of the box in height.
        width_factor (float): Factor by which to inflate each side of the box in width.
    Returns:
        dict: Dictionary with inflated 'boxes' and original 'labels'.
    """
    boxes = targets["boxes"]

    inflated_boxes = []

    # Inflate boxes
    for box in boxes:
        x_min, y_min, x_max, y_max = box

        # Inflate the box by the specified factor
        box_width = x_max - x_min
        box_height = y_max - y_min

        x_min = max(0, x_min - width_factor * box_width)
        y_min = max(0, y_min - height_factor * box_height)
        x_max = min(width, x_max + width_factor * box_width)
        y_max = min(height, y_max + height_factor * box_height)

        # Update the box with new coordinates
        inflated_boxes.append([x_min, y_min, x_max, y_max])
    
    inflated_boxes = torch.tensor(inflated_boxes)
    
    return {"boxes": inflated_boxes, "labels": targets["labels"], "scores": targets["scores"]}