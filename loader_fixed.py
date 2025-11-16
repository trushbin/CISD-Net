"""
Fixed Data Loader for 3D YOLO

Key features:
- Consistent label format using absolute coordinates
- Proper data augmentation with coordinate transformations
- Support for dynamic image sizes
- Validation of coordinates at each step
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from pathlib import Path
from coordinate_utils import (
    xywh2xyxy, xyxy2xywh, validate_boxes, clip_boxes
)


class YOLODataset(Dataset):
    """
    YOLO Dataset with proper coordinate handling.
    
    Label format:
    - During loading: class, x_center, y_center, width, height (normalized [0, 1])
    - After processing: class, x1, y1, x2, y2 (absolute pixels)
    
    This ensures consistency: always work with absolute coordinates internally.
    """
    
    def __init__(self, img_dir, label_dir, img_size=640, augment=True, 
                 max_objects=100, cache_images=False):
        """
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing YOLO format labels
            img_size: Target image size (int or tuple)
            augment: Whether to apply augmentation
            max_objects: Maximum number of objects per image
            cache_images: Whether to cache images in memory
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment
        self.max_objects = max_objects
        
        # Find all images
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + 
                               list(self.img_dir.glob('*.png')) +
                               list(self.img_dir.glob('*.jpeg')))
        
        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        # Cache for images if requested
        self.image_cache = {} if cache_images else None
        
        print(f"Found {len(self.img_files)} images in {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Get item with proper coordinate handling.
        
        Returns:
            img: Tensor of shape (3, H, W)
            target: Dict with:
                - 'labels': (max_objects, 1) - class labels
                - 'bboxes': (max_objects, 4) - boxes in xyxy format (absolute)
                - 'mask': (max_objects,) - valid object mask
        """
        img_path = self.img_files[idx]
        
        # Load image
        if self.image_cache is not None and idx in self.image_cache:
            img = self.image_cache[idx].copy()
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.image_cache is not None:
                self.image_cache[idx] = img.copy()
        
        orig_h, orig_w = img.shape[:2]
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = self._load_labels(label_path, orig_w, orig_h)
        
        # Apply augmentation
        if self.augment:
            img, labels = self._augment(img, labels)
        
        # Resize image and adjust coordinates
        img, labels = self._resize(img, labels, self.img_size)
        
        # Convert image to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Prepare target in consistent format
        target = self._prepare_target(labels)
        
        return img, target
    
    def _load_labels(self, label_path, img_w, img_h):
        """
        Load labels from YOLO format file.
        
        YOLO format: class x_center y_center width height (normalized)
        Output: List of [class, x1, y1, x2, y2] (absolute pixels)
        
        Args:
            label_path: Path to label file
            img_w: Original image width
            img_h: Original image height
            
        Returns:
            labels: numpy array of shape (N, 5) - [class, x1, y1, x2, y2] in absolute coords
        """
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                cls = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])
                
                # Convert normalized xywh to absolute xyxy
                x_center = x_center_norm * img_w
                y_center = y_center_norm * img_h
                w = w_norm * img_w
                h = h_norm * img_h
                
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                
                labels.append([cls, x1, y1, x2, y2])
        
        if len(labels) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        labels = np.array(labels, dtype=np.float32)
        
        # Validate and clip boxes
        boxes = torch.from_numpy(labels[:, 1:5])
        valid = validate_boxes(boxes, img_w, img_h, format='xyxy', tolerance=1.0)
        labels = labels[valid.numpy()]
        
        if len(labels) > 0:
            boxes = torch.from_numpy(labels[:, 1:5])
            boxes = clip_boxes(boxes, img_w, img_h, format='xyxy')
            labels[:, 1:5] = boxes.numpy()
        
        return labels
    
    def _augment(self, img, labels):
        """
        Apply data augmentation.
        
        Args:
            img: Image array (H, W, 3)
            labels: Labels array (N, 5) - [class, x1, y1, x2, y2] in absolute coords
            
        Returns:
            img: Augmented image
            labels: Augmented labels with adjusted coordinates
        """
        h, w = img.shape[:2]
        
        # Random horizontal flip
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            if len(labels) > 0:
                labels[:, 1] = w - labels[:, 1]  # Flip x1
                labels[:, 3] = w - labels[:, 3]  # Flip x2
                # Swap x1 and x2 after flip
                labels[:, [1, 3]] = labels[:, [3, 1]]
        
        # Random HSV augmentation
        if np.random.rand() < 0.5:
            img = self._augment_hsv(img)
        
        # Random brightness/contrast
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)    # Brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        
        return img, labels
    
    def _augment_hsv(self, img, hgain=0.015, sgain=0.7, vgain=0.4):
        """Apply HSV augmentation."""
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)
        
        return img
    
    def _resize(self, img, labels, target_size):
        """
        Resize image and adjust label coordinates.
        
        Args:
            img: Image array (H, W, 3)
            labels: Labels array (N, 5) - [class, x1, y1, x2, y2] in absolute coords
            target_size: Target size (H, W)
            
        Returns:
            img: Resized image
            labels: Labels with adjusted coordinates
        """
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = target_size
        
        # Compute scale factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        
        # Resize image
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Adjust label coordinates
        if len(labels) > 0:
            labels[:, 1] *= scale_w  # x1
            labels[:, 2] *= scale_h  # y1
            labels[:, 3] *= scale_w  # x2
            labels[:, 4] *= scale_h  # y2
            
            # Clip to image bounds
            boxes = torch.from_numpy(labels[:, 1:5])
            boxes = clip_boxes(boxes, target_w, target_h, format='xyxy')
            labels[:, 1:5] = boxes.numpy()
        
        return img, labels
    
    def _prepare_target(self, labels):
        """
        Prepare target dict with fixed-size tensors.
        
        Args:
            labels: numpy array of shape (N, 5) - [class, x1, y1, x2, y2]
            
        Returns:
            target: Dict with padded tensors
        """
        num_objects = len(labels)
        
        # Create fixed-size tensors
        target_labels = torch.zeros(self.max_objects, 1, dtype=torch.long)
        target_bboxes = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        target_mask = torch.zeros(self.max_objects, dtype=torch.bool)
        
        if num_objects > 0:
            num_objects = min(num_objects, self.max_objects)
            
            target_labels[:num_objects, 0] = torch.from_numpy(labels[:num_objects, 0]).long()
            target_bboxes[:num_objects] = torch.from_numpy(labels[:num_objects, 1:5])
            target_mask[:num_objects] = True
        
        return {
            'labels': target_labels,
            'bboxes': target_bboxes,
            'mask': target_mask
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch: List of (img, target) tuples
            
        Returns:
            imgs: Tensor of shape (B, 3, H, W)
            targets: Dict with batched tensors
        """
        imgs, targets = zip(*batch)
        
        # Stack images
        imgs = torch.stack(imgs, 0)
        
        # Stack targets
        batched_targets = {
            'labels': torch.stack([t['labels'] for t in targets], 0),
            'bboxes': torch.stack([t['bboxes'] for t in targets], 0),
            'mask': torch.stack([t['mask'] for t in targets], 0)
        }
        
        return imgs, batched_targets


def create_dataloader(img_dir, label_dir, batch_size=16, img_size=640, 
                      augment=True, num_workers=4, shuffle=True, pin_memory=True):
    """
    Create DataLoader for training or validation.
    
    Args:
        img_dir: Directory containing images
        label_dir: Directory containing labels
        batch_size: Batch size
        img_size: Image size
        augment: Whether to apply augmentation
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = YOLODataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=YOLODataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=True if augment else False
    )
    
    return dataloader


class InferenceDataset(Dataset):
    """Simple dataset for inference without labels."""
    
    def __init__(self, img_paths, img_size=640):
        self.img_paths = [Path(p) for p in img_paths]
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        
        # Resize
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), 
                        interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, str(img_path), (orig_w, orig_h)
