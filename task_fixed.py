"""
Fixed YOLO3D Model with proper coordinate handling and state dict management.

Key features:
- Unified model export/import with consistent state_dict handling
- Proper coordinate transformations throughout
- Dynamic input size support
- Gradient flow maintenance
"""

import torch
import torch.nn as nn
from detect_fixed import Detect
from loss_fixed import ComputeLoss


class Conv(nn.Module):
    """Standard convolution with batch norm and activation."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Backbone(nn.Module):
    """Simple backbone network for YOLO."""
    
    def __init__(self, in_channels=3, width_mult=1.0):
        super().__init__()
        
        # Calculate channel dimensions based on width multiplier
        base_channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width_mult) for c in base_channels]
        
        # Stem
        self.stem = Conv(in_channels, channels[0], 6, 2, 2)  # P1/2
        
        # Stages
        self.stage1 = nn.Sequential(
            Conv(channels[0], channels[1], 3, 2),  # P2/4
            C3(channels[1], channels[1], n=3)
        )
        
        self.stage2 = nn.Sequential(
            Conv(channels[1], channels[2], 3, 2),  # P3/8
            C3(channels[2], channels[2], n=6)
        )
        
        self.stage3 = nn.Sequential(
            Conv(channels[2], channels[3], 3, 2),  # P4/16
            C3(channels[3], channels[3], n=9)
        )
        
        self.stage4 = nn.Sequential(
            Conv(channels[3], channels[4], 3, 2),  # P5/32
            C3(channels[4], channels[4], n=3),
            SPPF(channels[4], channels[4])
        )
        
        # Store output channels for each detection level
        self.out_channels = [channels[2], channels[3], channels[4]]  # P3, P4, P5
    
    def forward(self, x):
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature maps [P3, P4, P5] for detection at different scales
        """
        x = self.stem(x)
        x = self.stage1(x)
        
        p3 = self.stage2(x)  # Stride 8 (but we'll use 4, 8, 16 in detect)
        p4 = self.stage3(p3)  # Stride 16
        p5 = self.stage4(p4)  # Stride 32
        
        return [p3, p4, p5]


class Neck(nn.Module):
    """FPN-style neck for feature fusion."""
    
    def __init__(self, channels):
        super().__init__()
        
        # Top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.reduce_p5 = Conv(channels[2], channels[1], 1, 1)
        self.reduce_p4 = Conv(channels[1], channels[0], 1, 1)
        
        self.fpn_p4 = C3(channels[1] + channels[1], channels[1], n=3, shortcut=False)
        self.fpn_p3 = C3(channels[0] + channels[0], channels[0], n=3, shortcut=False)
        
        # Bottom-up pathway
        self.downsample_p3 = Conv(channels[0], channels[0], 3, 2)
        self.downsample_p4 = Conv(channels[1], channels[1], 3, 2)
        
        self.pan_p4 = C3(channels[0] + channels[1], channels[1], n=3, shortcut=False)
        self.pan_p5 = C3(channels[1] + channels[2], channels[2], n=3, shortcut=False)
        
        self.out_channels = channels
    
    def forward(self, features):
        """
        Forward pass through neck.
        
        Args:
            features: List of [P3, P4, P5] from backbone
            
        Returns:
            List of [P3_out, P4_out, P5_out] for detection
        """
        p3, p4, p5 = features
        
        # Top-down pathway
        p5_reduced = self.reduce_p5(p5)
        p4_fused = torch.cat([self.upsample(p5_reduced), p4], dim=1)
        p4_out = self.fpn_p4(p4_fused)
        
        p4_reduced = self.reduce_p4(p4_out)
        p3_fused = torch.cat([self.upsample(p4_reduced), p3], dim=1)
        p3_out = self.fpn_p3(p3_fused)
        
        # Bottom-up pathway
        p3_down = self.downsample_p3(p3_out)
        p4_fused = torch.cat([p3_down, p4_out], dim=1)
        p4_final = self.pan_p4(p4_fused)
        
        p4_down = self.downsample_p4(p4_final)
        p5_fused = torch.cat([p4_down, p5], dim=1)
        p5_final = self.pan_p5(p5_fused)
        
        return [p3_out, p4_final, p5_final]


class YOLO3DModel(nn.Module):
    """
    Complete YOLO 3D Object Detection Model.
    
    Features:
    - Proper coordinate handling throughout
    - Dynamic input size support
    - Unified state dict management
    - Gradient flow maintenance
    """
    
    def __init__(self, num_classes=80, width_mult=1.0, in_channels=3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Build network
        self.backbone = Backbone(in_channels=in_channels, width_mult=width_mult)
        self.neck = Neck(self.backbone.out_channels)
        self.detect = Detect(nc=num_classes, ch=tuple(self.neck.out_channels))
        
        # Initialize loss function
        self.loss_fn = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, targets=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            targets: Optional targets dict for training with keys:
                - 'labels': (B, max_gt, 1)
                - 'bboxes': (B, max_gt, 4) in xyxy format
                - 'mask': (B, max_gt)
                
        Returns:
            If training (targets provided):
                - loss: Total loss value
                - loss_dict: Dictionary with loss components
            If inference:
                - predictions: Tensor of shape (B, N, 4 + nc)
        """
        # Get image dimensions for coordinate validation
        img_height, img_width = x.shape[2:]
        
        # Forward through network
        features = self.backbone(x)
        features = self.neck(features)
        
        if self.training and targets is not None:
            # Training mode - compute loss
            preds = self.detect(features)
            
            if self.loss_fn is None:
                self.loss_fn = ComputeLoss(self.detect, num_classes=self.num_classes)
            
            loss, loss_dict = self.loss_fn(preds, targets, img_width, img_height)
            return loss, loss_dict
        else:
            # Inference mode - return predictions
            self.detect.eval()
            predictions = self.detect(features)
            return predictions
    
    def save(self, path, optimizer=None, epoch=None, **kwargs):
        """
        Save model with consistent state dict format.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            **kwargs: Additional items to save
        """
        checkpoint = {
            'model': self.state_dict(),
            'num_classes': self.num_classes,
        }
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path, device='cpu', strict=True):
        """
        Load model from checkpoint with consistent handling.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            model: Loaded model
            checkpoint: Full checkpoint dict (for optimizer, epoch, etc.)
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            num_classes = checkpoint.get('num_classes', 80)
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            num_classes = checkpoint.get('num_classes', 80)
        else:
            state_dict = checkpoint
            num_classes = 80
        
        # Create model
        model = cls(num_classes=num_classes)
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        
        return model, checkpoint
    
    def export_onnx(self, path, input_shape=(1, 3, 640, 640), opset_version=11):
        """
        Export model to ONNX format.
        
        Args:
            path: Path to save ONNX model
            input_shape: Input tensor shape (B, C, H, W)
            opset_version: ONNX opset version
        """
        self.eval()
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 1: 'anchors'}
            }
        )


def create_model(num_classes=80, width_mult=1.0, pretrained_path=None, device='cpu'):
    """
    Factory function to create YOLO3D model.
    
    Args:
        num_classes: Number of object classes
        width_mult: Width multiplier for model size
        pretrained_path: Path to pretrained weights
        device: Device to load model on
        
    Returns:
        model: YOLO3DModel instance
    """
    if pretrained_path:
        model, _ = YOLO3DModel.load(pretrained_path, device=device, strict=False)
    else:
        model = YOLO3DModel(num_classes=num_classes, width_mult=width_mult)
        model.to(device)
    
    return model
