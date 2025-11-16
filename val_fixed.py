"""
Fixed Validation Script for 3D YOLO

Key features:
- Correct coordinate decoding
- NMS (Non-Maximum Suppression) with proper IoU computation
- Support for various evaluation metrics
- Visualization utilities
"""

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from task_fixed import YOLO3DModel, create_model
from loader_fixed import create_dataloader, InferenceDataset
from coordinate_utils import box_iou, clip_boxes, xywh2xyxy


def non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45, 
                        max_det=300, nc=80):
    """
    Perform Non-Maximum Suppression (NMS) on predictions.
    
    Args:
        predictions: Tensor of shape (B, N, 4 + nc)
                    First 4 are bbox in xyxy format, rest are class scores
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum number of detections per image
        nc: Number of classes
        
    Returns:
        List of detections per image, each of shape (M, 6)
        [x1, y1, x2, y2, conf, class]
    """
    batch_size = predictions.shape[0]
    output = []
    
    for img_idx in range(batch_size):
        pred = predictions[img_idx]  # (N, 4 + nc)
        
        # Get boxes and scores
        boxes = pred[:, :4]  # (N, 4) - xyxy format
        scores = pred[:, 4:]  # (N, nc)
        
        # Get max score and class for each prediction
        class_conf, class_idx = scores.max(dim=1)  # (N,)
        
        # Filter by confidence
        conf_mask = class_conf >= conf_thres
        
        if not conf_mask.any():
            output.append(torch.zeros((0, 6), device=pred.device))
            continue
        
        # Apply mask
        boxes = boxes[conf_mask]  # (M, 4)
        class_conf = class_conf[conf_mask]  # (M,)
        class_idx = class_idx[conf_mask]  # (M,)
        
        # NMS per class
        unique_classes = class_idx.unique()
        keep_indices = []
        
        for cls in unique_classes:
            cls_mask = class_idx == cls
            cls_boxes = boxes[cls_mask]
            cls_conf = class_conf[cls_mask]
            cls_idx_in_filtered = torch.where(cls_mask)[0]
            
            # Sort by confidence
            sorted_idx = cls_conf.argsort(descending=True)
            cls_boxes = cls_boxes[sorted_idx]
            cls_idx_sorted = cls_idx_in_filtered[sorted_idx]
            
            # Apply NMS
            keep_cls = []
            while cls_boxes.shape[0] > 0:
                # Keep highest confidence detection
                keep_cls.append(cls_idx_sorted[0].item())
                
                if cls_boxes.shape[0] == 1:
                    break
                
                # Compute IoU with remaining boxes
                iou = box_iou(cls_boxes[0:1], cls_boxes[1:], format='xyxy')
                
                # Keep boxes with IoU below threshold
                mask = iou.squeeze() < iou_thres
                cls_boxes = cls_boxes[1:][mask]
                cls_idx_sorted = cls_idx_sorted[1:][mask]
            
            keep_indices.extend(keep_cls)
        
        # Get final detections
        if keep_indices:
            keep_indices = torch.tensor(keep_indices, device=pred.device)
            final_boxes = boxes[keep_indices]
            final_conf = class_conf[keep_indices]
            final_cls = class_idx[keep_indices]
            
            # Combine into output format
            detections = torch.cat([
                final_boxes,
                final_conf.unsqueeze(1),
                final_cls.unsqueeze(1).float()
            ], dim=1)  # (K, 6)
            
            # Limit number of detections
            if detections.shape[0] > max_det:
                # Keep top max_det by confidence
                top_idx = final_conf.argsort(descending=True)[:max_det]
                detections = detections[top_idx]
            
            output.append(detections)
        else:
            output.append(torch.zeros((0, 6), device=pred.device))
    
    return output


def scale_boxes(boxes, img_shape, orig_shape):
    """
    Scale boxes from current image shape to original image shape.
    
    Args:
        boxes: (N, 4) in xyxy format
        img_shape: Current image shape (H, W)
        orig_shape: Original image shape (H, W)
        
    Returns:
        Scaled boxes
    """
    # Calculate scale factors
    scale_h = orig_shape[0] / img_shape[0]
    scale_w = orig_shape[1] / img_shape[1]
    
    # Scale coordinates
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    
    # Clip to original image bounds
    boxes = clip_boxes(boxes, orig_shape[1], orig_shape[0], format='xyxy')
    
    return boxes


class Validator:
    """Validation manager for YOLO3D."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {args.weights}...")
        self.model, _ = YOLO3DModel.load(args.weights, device=self.device)
        self.model.eval()
        
        print(f"Model loaded with {self.model.num_classes} classes")
        
        # Create output directory
        if args.save_dir:
            self.save_dir = Path(args.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
    
    def validate_dataset(self):
        """Validate on a dataset with labels."""
        # Create dataloader
        print(f"Loading validation data from {self.args.img_dir}...")
        val_loader = create_dataloader(
            img_dir=self.args.img_dir,
            label_dir=self.args.label_dir,
            batch_size=self.args.batch_size,
            img_size=self.args.img_size,
            augment=False,
            num_workers=self.args.num_workers,
            shuffle=False
        )
        
        # Validation loop
        print("Running validation...")
        pbar = tqdm(val_loader, desc="Validating")
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        with torch.no_grad():
            for images, targets in pbar:
                images = images.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Apply NMS
                detections = non_max_suppression(
                    predictions,
                    conf_thres=self.args.conf_thres,
                    iou_thres=self.args.iou_thres,
                    nc=self.model.num_classes
                )
                
                # Compute metrics (simplified - true/false positives)
                for i, dets in enumerate(detections):
                    gt_boxes = targets['bboxes'][i][targets['mask'][i]]
                    
                    if len(dets) > 0 and len(gt_boxes) > 0:
                        # Compute IoU between detections and ground truth
                        iou = box_iou(dets[:, :4], gt_boxes, format='xyxy')
                        
                        # Count matches (IoU > 0.5)
                        matches = (iou > 0.5).any(dim=1).sum().item()
                        total_tp += matches
                        total_fp += len(dets) - matches
                        total_fn += len(gt_boxes) - matches
                    elif len(dets) > 0:
                        total_fp += len(dets)
                    elif len(gt_boxes) > 0:
                        total_fn += len(gt_boxes)
                
                # Update progress bar
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                
                pbar.set_postfix({
                    'P': f"{precision:.3f}",
                    'R': f"{recall:.3f}"
                })
        
        # Final metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nValidation Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  True Positives: {total_tp}")
        print(f"  False Positives: {total_fp}")
        print(f"  False Negatives: {total_fn}")
    
    def infer_images(self):
        """Run inference on images without labels."""
        # Get image paths
        img_dir = Path(self.args.img_dir)
        img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        print(f"Found {len(img_paths)} images")
        
        # Process images
        for img_path in tqdm(img_paths, desc="Processing images"):
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            orig_h, orig_w = img.shape[:2]
            
            # Resize
            img_resized = cv2.resize(img, (self.args.img_size, self.args.img_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
            
            # Apply NMS
            detections = non_max_suppression(
                predictions,
                conf_thres=self.args.conf_thres,
                iou_thres=self.args.iou_thres,
                nc=self.model.num_classes
            )[0]
            
            # Scale boxes back to original image size
            if len(detections) > 0:
                detections[:, :4] = scale_boxes(
                    detections[:, :4],
                    (self.args.img_size, self.args.img_size),
                    (orig_h, orig_w)
                )
            
            # Visualize if requested
            if self.args.visualize and self.save_dir:
                self._visualize(img, detections, img_path.name)
            
            # Save detections
            if self.save_dir:
                self._save_detections(detections, img_path.stem)
        
        print(f"Results saved to {self.save_dir}")
    
    def _visualize(self, img, detections, img_name):
        """Draw detections on image."""
        img_vis = img.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            
            # Draw box
            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Draw label
            label = f"cls{int(cls)}: {conf:.2f}"
            cv2.putText(img_vis, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save
        save_path = self.save_dir / f"vis_{img_name}"
        cv2.imwrite(str(save_path), img_vis)
    
    def _save_detections(self, detections, img_name):
        """Save detections to text file."""
        save_path = self.save_dir / f"{img_name}.txt"
        
        with open(save_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                f.write(f"{int(cls)} {conf:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate YOLO3D Model')
    
    # Model
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    
    # Data
    parser.add_argument('--img-dir', type=str, required=True,
                       help='Images directory')
    parser.add_argument('--label-dir', type=str, default=None,
                       help='Labels directory (for validation with GT)')
    
    # Inference
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='./runs/val',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize detections')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    return parser.parse_args()


def main():
    """Main validation function."""
    args = parse_args()
    
    validator = Validator(args)
    
    if args.label_dir:
        # Validation with ground truth
        validator.validate_dataset()
    else:
        # Inference only
        validator.infer_images()


if __name__ == '__main__':
    main()
