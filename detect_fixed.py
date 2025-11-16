"""
Fixed Detect Module for 3D YOLO

Key fixes:
- Correct stride initialization: [4, 8, 16] for 3 detection levels
- Dynamic input size support (no hardcoded dimensions)
- Proper DFL (Distribution Focal Loss) module with correct buffer dimensions
- Coordinate decoding using absolute coordinates
"""

import torch
import torch.nn as nn
import math
from coordinate_utils import dist2bbox, make_anchors


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    Converts distribution predictions to box distance predictions.
    
    Args:
        c1: Number of input channels (typically reg_max * 4)
        reg_max: Maximum regression distance (default: 16)
    """
    
    def __init__(self, c1=64, reg_max=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 4, 1, bias=False)
        self.reg_max = reg_max
        
        # Create projection matrix with correct dimensions
        # Shape: (reg_max,) - represents weights for each distance level
        self.register_buffer('proj', torch.arange(reg_max, dtype=torch.float))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, reg_max * 4, H, W)
            
        Returns:
            Distance predictions of shape (B, 4, H, W)
        """
        b, c, h, w = x.shape
        
        # Reshape to (B, 4, reg_max, H, W)
        x = x.view(b, 4, self.reg_max, h, w)
        
        # Apply softmax over reg_max dimension to get distribution
        x = x.softmax(2)
        
        # Project to get expected distance: sum(prob_i * distance_i)
        # x: (B, 4, reg_max, H, W), proj: (reg_max,)
        x = (x * self.proj.view(1, 1, -1, 1, 1)).sum(2)
        
        return x


class Detect(nn.Module):
    """
    YOLO Detection Head with proper coordinate handling.
    
    Key features:
    - Three detection levels with strides [4, 8, 16]
    - Dynamic input size support
    - DFL for better box regression
    - Outputs in absolute coordinates
    """
    
    def __init__(self, nc=80, ch=(256, 512, 1024), reg_max=16):
        """
        Args:
            nc: Number of classes
            ch: Tuple of channel sizes for each detection level
            reg_max: Maximum regression distance for DFL
        """
        super().__init__()
        
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers (3)
        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        
        # FIXED: Correct stride initialization for 3 detection levels
        self.stride = torch.tensor([4.0, 8.0, 16.0])  # NOT [8, 16]
        
        # Create detection heads for each level
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, max(64, x // 4), 3, padding=1),
                nn.Conv2d(max(64, x // 4), max(64, x // 4), 3, padding=1),
                nn.Conv2d(max(64, x // 4), 4 * self.reg_max, 1)
            ) for x in ch
        )
        
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, max(64, x // 4), 3, padding=1),
                nn.Conv2d(max(64, x // 4), max(64, x // 4), 3, padding=1),
                nn.Conv2d(max(64, x // 4), self.nc, 1)
            ) for x in ch
        )
        
        self.dfl = DFL(self.reg_max * 4, reg_max=self.reg_max)
        
        # Initialize weights
        self._initialize_biases()
    
    def _initialize_biases(self):
        """Initialize biases for detection heads."""
        for cv in self.cv3:
            # Initialize classification bias
            b = cv[-1].bias.view(-1)
            b.data.fill_(-math.log((1 - 0.01) / 0.01))
            cv[-1].bias = nn.Parameter(b.view(-1), requires_grad=True)
    
    def forward(self, x):
        """
        Forward pass through detection head.
        
        Args:
            x: List of feature maps from backbone [P3, P4, P5]
               Each with shape (B, C, H, W)
               
        Returns:
            If training:
                Tuple of (box_preds, cls_preds, feat_maps)
                - box_preds: List of box predictions for each level
                - cls_preds: List of class predictions for each level
                - feat_maps: Original feature maps (for anchor generation)
            If inference:
                Concatenated predictions of shape (B, N, 4 + nc)
                where N is total number of anchors across all levels
        """
        assert len(x) == self.nl, f"Expected {self.nl} feature maps, got {len(x)}"
        
        box_preds = []
        cls_preds = []
        
        for i in range(self.nl):
            # Box predictions: (B, 4 * reg_max, H, W)
            box_pred = self.cv2[i](x[i])
            
            # Class predictions: (B, nc, H, W)
            cls_pred = self.cv3[i](x[i])
            
            box_preds.append(box_pred)
            cls_preds.append(cls_pred)
        
        if self.training:
            # Return raw predictions for loss computation
            return box_preds, cls_preds, x
        else:
            # Decode predictions for inference
            return self.decode_predictions(box_preds, cls_preds, x)
    
    def decode_predictions(self, box_preds, cls_preds, feats):
        """
        Decode predictions to absolute coordinates for inference.
        
        Args:
            box_preds: List of box predictions (B, 4 * reg_max, H, W) for each level
            cls_preds: List of class predictions (B, nc, H, W) for each level
            feats: List of feature maps for anchor generation
            
        Returns:
            Tensor of shape (B, N, 4 + nc) with decoded predictions
            First 4 channels are bbox in xyxy format (absolute coordinates)
            Last nc channels are class probabilities
        """
        device = box_preds[0].device
        batch_size = box_preds[0].shape[0]
        
        # Ensure stride is on the correct device
        if self.stride.device != device:
            self.stride = self.stride.to(device)
        
        # Generate anchors dynamically based on feature map sizes
        anchor_points, stride_tensor = make_anchors(feats, self.stride)
        
        decoded_boxes = []
        decoded_classes = []
        
        for i, (box_pred, cls_pred) in enumerate(zip(box_preds, cls_preds)):
            b, _, h, w = box_pred.shape
            
            # Apply DFL to get distances
            distances = self.dfl(box_pred)  # (B, 4, H, W)
            
            # Reshape to (B, 4, H * W) then (B, H * W, 4)
            distances = distances.view(b, 4, -1).permute(0, 2, 1)
            
            # Get anchor points for this level
            num_anchors_level = h * w
            start_idx = sum([feats[j].shape[2] * feats[j].shape[3] for j in range(i)])
            anchors = anchor_points[start_idx:start_idx + num_anchors_level].unsqueeze(0)  # (1, N, 2)
            anchors = anchors.expand(b, -1, -1)  # (B, N, 2)
            
            # Decode boxes to absolute coordinates (xyxy format)
            boxes = dist2bbox(distances, anchors, xywh=False)  # (B, N, 4)
            
            # Process class predictions
            cls_pred = cls_pred.view(b, self.nc, -1).permute(0, 2, 1)  # (B, H * W, nc)
            cls_pred = cls_pred.sigmoid()
            
            decoded_boxes.append(boxes)
            decoded_classes.append(cls_pred)
        
        # Concatenate all levels
        all_boxes = torch.cat(decoded_boxes, dim=1)  # (B, N_total, 4)
        all_classes = torch.cat(decoded_classes, dim=1)  # (B, N_total, nc)
        
        # Combine boxes and classes
        predictions = torch.cat([all_boxes, all_classes], dim=-1)  # (B, N_total, 4 + nc)
        
        return predictions


class DetectMultiBackend(nn.Module):
    """
    Wrapper for Detect module with backend support.
    Can be extended to support ONNX, TensorRT, etc.
    """
    
    def __init__(self, weights=None, device=None, nc=80, ch=(256, 512, 1024)):
        super().__init__()
        
        self.device = device if device else torch.device('cpu')
        self.detect = Detect(nc=nc, ch=ch).to(self.device)
        
        if weights:
            self.load_weights(weights)
    
    def load_weights(self, weights_path):
        """Load model weights from file."""
        ckpt = torch.load(weights_path, map_location=self.device)
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        self.detect.load_state_dict(state_dict, strict=False)
    
    def forward(self, x):
        return self.detect(x)
