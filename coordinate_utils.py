"""
Coordinate Utilities for 3D YOLO Object Detection

This module provides utility functions for coordinate transformations,
ensuring consistency throughout the framework by using absolute coordinates.

Key principles:
- Always use absolute coordinates (pixels) internally
- Clear conversion functions when normalization is needed (e.g., for loss)
- Validation functions to ensure coordinates are within bounds
"""

import torch
import torch.nn.functional as F


def xyxy2xywh(boxes):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height] format.
    All coordinates are in absolute pixel values.
    
    Args:
        boxes: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        
    Returns:
        Tensor of shape (..., 4) in [x_center, y_center, width, height] format
    """
    xy1 = boxes[..., :2]
    xy2 = boxes[..., 2:4]
    
    wh = xy2 - xy1
    xy = (xy1 + xy2) / 2
    
    return torch.cat([xy, wh], dim=-1)


def xywh2xyxy(boxes):
    """
    Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2] format.
    All coordinates are in absolute pixel values.
    
    Args:
        boxes: Tensor of shape (..., 4) in [x_center, y_center, width, height] format
        
    Returns:
        Tensor of shape (..., 4) in [x1, y1, x2, y2] format
    """
    xy = boxes[..., :2]
    wh = boxes[..., 2:4]
    
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    
    return torch.cat([xy1, xy2], dim=-1)


def normalize_coordinates(boxes, img_width, img_height):
    """
    Normalize absolute coordinates to [0, 1] range.
    Used when needed for specific operations (e.g., some loss functions).
    
    Args:
        boxes: Tensor of shape (..., 4) with absolute coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tensor of shape (..., 4) with normalized coordinates [0, 1]
    """
    boxes = boxes.clone()
    boxes[..., [0, 2]] /= img_width
    boxes[..., [1, 3]] /= img_height
    return boxes


def denormalize_coordinates(boxes, img_width, img_height):
    """
    Denormalize coordinates from [0, 1] range to absolute pixels.
    
    Args:
        boxes: Tensor of shape (..., 4) with normalized coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tensor of shape (..., 4) with absolute coordinates
    """
    boxes = boxes.clone()
    boxes[..., [0, 2]] *= img_width
    boxes[..., [1, 3]] *= img_height
    return boxes


def clip_boxes(boxes, img_width, img_height, format='xyxy'):
    """
    Clip bounding boxes to image boundaries.
    
    Args:
        boxes: Tensor of shape (..., 4)
        img_width: Image width in pixels
        img_height: Image height in pixels
        format: 'xyxy' or 'xywh'
        
    Returns:
        Clipped boxes tensor
    """
    if format == 'xyxy':
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img_width)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img_height)
    elif format == 'xywh':
        # Convert to xyxy, clip, then convert back
        boxes_xyxy = xywh2xyxy(boxes)
        boxes_xyxy[..., [0, 2]] = boxes_xyxy[..., [0, 2]].clamp(0, img_width)
        boxes_xyxy[..., [1, 3]] = boxes_xyxy[..., [1, 3]].clamp(0, img_height)
        boxes = xyxy2xywh(boxes_xyxy)
    
    return boxes


def validate_boxes(boxes, img_width, img_height, format='xyxy', tolerance=1e-3):
    """
    Validate that boxes are within image boundaries.
    
    Args:
        boxes: Tensor of shape (..., 4)
        img_width: Image width in pixels
        img_height: Image height in pixels
        format: 'xyxy' or 'xywh'
        tolerance: Small tolerance for floating point errors
        
    Returns:
        Boolean tensor indicating valid boxes
    """
    if format == 'xyxy':
        valid = (
            (boxes[..., 0] >= -tolerance) &
            (boxes[..., 1] >= -tolerance) &
            (boxes[..., 2] <= img_width + tolerance) &
            (boxes[..., 3] <= img_height + tolerance) &
            (boxes[..., 2] > boxes[..., 0]) &
            (boxes[..., 3] > boxes[..., 1])
        )
    elif format == 'xywh':
        boxes_xyxy = xywh2xyxy(boxes)
        valid = validate_boxes(boxes_xyxy, img_width, img_height, format='xyxy', tolerance=tolerance)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return valid


def box_area(boxes, format='xyxy'):
    """
    Compute area of bounding boxes.
    
    Args:
        boxes: Tensor of shape (..., 4)
        format: 'xyxy' or 'xywh'
        
    Returns:
        Tensor of shape (...,) with box areas
    """
    if format == 'xyxy':
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    elif format == 'xywh':
        return boxes[..., 2] * boxes[..., 3]
    else:
        raise ValueError(f"Unknown format: {format}")


def box_iou(boxes1, boxes2, format='xyxy'):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)
        format: 'xyxy' or 'xywh'
        
    Returns:
        Tensor of shape (N, M) with IoU values
    """
    if format == 'xywh':
        boxes1 = xywh2xyxy(boxes1)
        boxes2 = xywh2xyxy(boxes2)
    
    area1 = box_area(boxes1, format='xyxy')
    area2 = box_area(boxes2, format='xyxy')
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)
    
    union = area1[:, None] + area2[None, :] - inter
    
    iou = inter / (union + 1e-7)
    return iou


def dist2bbox(distance, anchor_points, xywh=True):
    """
    Transform distance predictions to bounding boxes.
    Used in YOLO detection heads with DFL (Distribution Focal Loss).
    
    Args:
        distance: Tensor of shape (N, 4) representing [left, top, right, bottom] distances
        anchor_points: Tensor of shape (N, 2) representing anchor point coordinates
        xywh: If True, return in xywh format; otherwise xyxy format
        
    Returns:
        Bounding boxes in absolute coordinates
    """
    lt = anchor_points - distance[..., :2]
    rb = anchor_points + distance[..., 2:]
    
    boxes = torch.cat([lt, rb], dim=-1)
    
    if xywh:
        boxes = xyxy2xywh(boxes)
    
    return boxes


def bbox2dist(boxes, anchor_points, reg_max=16):
    """
    Transform bounding boxes to distance predictions for training.
    
    Args:
        boxes: Tensor of shape (N, 4) in xyxy format with absolute coordinates
        anchor_points: Tensor of shape (N, 2) representing anchor point coordinates
        reg_max: Maximum regression distance (typically 16)
        
    Returns:
        Distance predictions of shape (N, 4)
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes with 4 coordinates, got {boxes.shape[-1]}")
    
    lt = anchor_points - boxes[..., :2]
    rb = boxes[..., 2:] - anchor_points
    
    distances = torch.cat([lt, rb], dim=-1).clamp(0, reg_max - 0.01)
    
    return distances


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchor points and stride tensors for YOLO detection.
    
    Args:
        feats: List of feature maps from different detection levels
        strides: List of stride values [4, 8, 16] for each detection level
        grid_cell_offset: Offset for anchor points (0.5 for center)
        
    Returns:
        anchor_points: Tensor of shape (N, 2) with absolute anchor coordinates
        stride_tensor: Tensor of shape (N, 1) with corresponding strides
    """
    anchor_points = []
    stride_tensor = []
    
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        
        # Create grid coordinates
        shifts_x = (torch.arange(w, device=feat.device) + grid_cell_offset) * stride
        shifts_y = (torch.arange(h, device=feat.device) + grid_cell_offset) * stride
        
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
        anchor_points.append(anchor_point)
        
        stride_tensor.append(torch.full((anchor_point.shape[0], 1), stride, 
                                       dtype=feat.dtype, device=feat.device))
    
    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0)
    
    return anchor_points, stride_tensor
