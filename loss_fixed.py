"""
Fixed Loss Module for 3D YOLO

Key fixes:
- No hardcoded image dimensions - supports dynamic input sizes
- Task-aligned assigner for better positive/negative sample selection
- Proper gradient flow maintenance
- Coordinate validation at each step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from coordinate_utils import (
    bbox2dist, box_iou, xywh2xyxy, xyxy2xywh,
    validate_boxes, make_anchors
)


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for YOLO.
    Assigns ground truth boxes to anchors based on both classification and localization quality.
    
    Key features:
    - Handles dynamic image sizes
    - Task-aligned assignment (considers both cls and iou)
    - Topk selection for positive samples
    """
    
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, 
                mask_gt, img_width, img_height):
        """
        Assign ground truth boxes to predictions.
        
        Args:
            pred_scores: (B, N, nc) - predicted class scores
            pred_bboxes: (B, N, 4) - predicted boxes in xyxy format (absolute coords)
            anchor_points: (N, 2) - anchor point coordinates (absolute)
            gt_labels: (B, max_gt, 1) - ground truth class labels
            gt_bboxes: (B, max_gt, 4) - ground truth boxes in xyxy format (absolute)
            mask_gt: (B, max_gt) - mask for valid ground truth boxes
            img_width: Image width for validation
            img_height: Image height for validation
            
        Returns:
            target_labels: (B, N) - assigned class labels
            target_bboxes: (B, N, 4) - assigned boxes
            target_scores: (B, N, nc) - target scores for training
            fg_mask: (B, N) - foreground mask
        """
        batch_size, num_anchors = pred_scores.shape[:2]
        num_max_boxes = gt_bboxes.shape[1]
        
        # Initialize outputs
        device = pred_scores.device
        target_labels = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(batch_size, num_anchors, 4, device=device)
        target_scores = torch.zeros(batch_size, num_anchors, self.num_classes, device=device)
        fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)
        
        # Process each image in batch
        for batch_idx in range(batch_size):
            num_gt = mask_gt[batch_idx].sum()
            if num_gt == 0:
                continue
            
            # Get valid ground truth boxes for this image
            gt_bbox = gt_bboxes[batch_idx, :num_gt]  # (num_gt, 4)
            gt_label = gt_labels[batch_idx, :num_gt].squeeze(-1)  # (num_gt,)
            
            # Validate ground truth boxes
            valid_gt = validate_boxes(gt_bbox, img_width, img_height, format='xyxy')
            if not valid_gt.all():
                # Skip invalid ground truth boxes
                gt_bbox = gt_bbox[valid_gt]
                gt_label = gt_label[valid_gt]
                num_gt = gt_bbox.shape[0]
                if num_gt == 0:
                    continue
            
            # Get predictions for this image
            pred_score = pred_scores[batch_idx]  # (N, nc)
            pred_bbox = pred_bboxes[batch_idx]  # (N, 4)
            
            # Compute IoU between predictions and ground truth
            iou = box_iou(pred_bbox, gt_bbox, format='xyxy')  # (N, num_gt)
            
            # Get classification scores for ground truth classes
            # Use one-hot encoding to select relevant class scores
            gt_label_onehot = F.one_hot(gt_label, self.num_classes).float()  # (num_gt, nc)
            pred_score_gt = (pred_score.unsqueeze(1) * gt_label_onehot.unsqueeze(0)).sum(-1)  # (N, num_gt)
            
            # Compute alignment metric (task-aligned)
            # Combines classification score and IoU quality
            alignment_metric = pred_score_gt.sigmoid().pow(self.alpha) * iou.pow(self.beta)
            
            # Select topk candidates for each ground truth
            topk_metric, topk_idxs = alignment_metric.topk(self.topk, dim=0, largest=True)
            
            # Create mask for positive samples
            is_in_topk = torch.zeros_like(alignment_metric, dtype=torch.bool)
            is_in_topk.scatter_(0, topk_idxs, True)
            
            # Filter by IoU threshold
            is_pos = is_in_topk & (iou > 0.0)
            
            # Handle case where multiple GTs are assigned to same anchor
            # Keep the GT with highest alignment metric
            for anchor_idx in range(num_anchors):
                if is_pos[anchor_idx].any():
                    max_metric_idx = alignment_metric[anchor_idx][is_pos[anchor_idx]].argmax()
                    gt_idx = torch.where(is_pos[anchor_idx])[0][max_metric_idx]
                    
                    # Assign to this anchor
                    target_labels[batch_idx, anchor_idx] = gt_label[gt_idx]
                    target_bboxes[batch_idx, anchor_idx] = gt_bbox[gt_idx]
                    target_scores[batch_idx, anchor_idx, gt_label[gt_idx]] = alignment_metric[anchor_idx, gt_idx]
                    fg_mask[batch_idx, anchor_idx] = True
        
        return target_labels, target_bboxes, target_scores, fg_mask


class BboxLoss(nn.Module):
    """
    Bounding Box Loss combining IoU loss and DFL loss.
    Supports dynamic image sizes.
    """
    
    def __init__(self, reg_max=16, use_dfl=True):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
    
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, 
                target_scores, target_scores_sum, fg_mask):
        """
        Compute bounding box loss.
        
        Args:
            pred_dist: (B, N, 4 * reg_max) - predicted distributions
            pred_bboxes: (B, N, 4) - predicted boxes in xyxy format
            anchor_points: (N, 2) - anchor points
            target_bboxes: (B, N, 4) - target boxes in xyxy format
            target_scores: (B, N) - target quality scores
            target_scores_sum: Scalar - sum of target scores for normalization
            fg_mask: (B, N) - foreground mask
            
        Returns:
            loss_iou: IoU loss
            loss_dfl: DFL loss
        """
        # Select foreground samples
        fg_mask = fg_mask.unsqueeze(-1)
        
        # Compute IoU loss
        weight = target_scores.sum(-1)[fg_mask.squeeze(-1)]
        iou = bbox_iou(pred_bboxes[fg_mask.squeeze(-1)], 
                       target_bboxes[fg_mask.squeeze(-1)], 
                       format='xyxy')
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum.clamp(min=1.0)
        
        # Compute DFL loss if enabled
        loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        if self.use_dfl:
            # Convert target boxes to distances
            target_ltrb = bbox2dist(target_bboxes[fg_mask.squeeze(-1)], 
                                   anchor_points.unsqueeze(0).expand(pred_bboxes.shape[0], -1, -1)[fg_mask.squeeze(-1)],
                                   reg_max=self.reg_max)
            
            # Reshape pred_dist for DFL computation
            pred_dist_fg = pred_dist[fg_mask.squeeze(-1)].view(-1, 4, self.reg_max)
            target_ltrb = target_ltrb.view(-1, 4)
            
            # DFL loss - essentially cross-entropy on the distance distribution
            loss_dfl = self._df_loss(pred_dist_fg, target_ltrb)
            loss_dfl = (loss_dfl * weight.unsqueeze(-1)).sum() / target_scores_sum.clamp(min=1.0)
        
        return loss_iou, loss_dfl
    
    def _df_loss(self, pred_dist, target):
        """
        Distribution Focal Loss.
        
        Args:
            pred_dist: (N, 4, reg_max) - predicted distributions
            target: (N, 4) - target distances
            
        Returns:
            loss: (N, 4) - DFL loss
        """
        # Clip target to valid range
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        
        # Get integer and fractional parts
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        # Compute cross-entropy style loss
        loss_left = F.cross_entropy(
            pred_dist.reshape(-1, self.reg_max),
            target_left.reshape(-1),
            reduction='none'
        ).reshape(-1, 4) * weight_left
        
        loss_right = F.cross_entropy(
            pred_dist.reshape(-1, self.reg_max),
            target_right.reshape(-1).clamp(max=self.reg_max - 1),
            reduction='none'
        ).reshape(-1, 4) * weight_right
        
        return loss_left + loss_right


def bbox_iou(box1, box2, format='xyxy', GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between boxes with optional GIoU, DIoU, CIoU.
    
    Args:
        box1: (N, 4)
        box2: (N, 4)
        format: 'xyxy' or 'xywh'
        
    Returns:
        IoU values: (N,)
    """
    if format == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)
    
    # Get the coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        # Convex width and height
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - 
                                                       torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    
    return iou


class ComputeLoss:
    """
    Complete loss computation for YOLO training.
    Handles dynamic image sizes and maintains proper gradient flow.
    """
    
    def __init__(self, model, use_dfl=True, reg_max=16, num_classes=80):
        self.device = next(model.parameters()).device
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.num_classes = num_classes
        
        # Loss components
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(reg_max=reg_max, use_dfl=use_dfl)
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)
        
        # Loss weights
        self.hyp_box = 7.5
        self.hyp_cls = 0.5
        self.hyp_dfl = 1.5
        
        # Get stride from model
        self.stride = model.stride
    
    def __call__(self, preds, targets, img_width, img_height):
        """
        Compute total loss.
        
        Args:
            preds: Tuple of (box_preds, cls_preds, feats)
                - box_preds: List of (B, 4 * reg_max, H, W) for each level
                - cls_preds: List of (B, nc, H, W) for each level
                - feats: List of feature maps for anchor generation
            targets: Dict with keys:
                - 'labels': (B, max_gt, 1) - class labels
                - 'bboxes': (B, max_gt, 4) - boxes in xyxy format (absolute)
                - 'mask': (B, max_gt) - valid gt mask
            img_width: Current image width (can vary per batch with dynamic sizing)
            img_height: Current image height
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        box_preds, cls_preds, feats = preds
        
        batch_size = cls_preds[0].shape[0]
        device = cls_preds[0].device
        
        # Generate anchors dynamically based on current feature map sizes
        anchor_points, stride_tensor = make_anchors(feats, self.stride.to(device))
        
        # Decode box predictions
        pred_bboxes, pred_scores, pred_distri = self._decode_preds(
            box_preds, cls_preds, anchor_points
        )
        
        # Get targets
        gt_labels = targets['labels'].to(device)  # (B, max_gt, 1)
        gt_bboxes = targets['bboxes'].to(device)  # (B, max_gt, 4)
        mask_gt = targets['mask'].to(device)  # (B, max_gt)
        
        # Assign targets to predictions
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach(),
            pred_bboxes.detach(),
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
            img_width,
            img_height
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Classification loss
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        
        # Box loss
        if fg_mask.sum():
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask
            )
        else:
            loss_iou = torch.tensor(0.0, device=device)
            loss_dfl = torch.tensor(0.0, device=device)
        
        # Total loss
        loss_cls *= self.hyp_cls
        loss_iou *= self.hyp_box
        loss_dfl *= self.hyp_dfl
        
        total_loss = loss_cls + loss_iou + loss_dfl
        
        return total_loss, {
            'loss': total_loss.item(),
            'loss_cls': loss_cls.item(),
            'loss_iou': loss_iou.item(),
            'loss_dfl': loss_dfl.item()
        }
    
    def _decode_preds(self, box_preds, cls_preds, anchor_points):
        """
        Decode predictions for loss computation.
        
        Returns:
            pred_bboxes: (B, N, 4) in xyxy format
            pred_scores: (B, N, nc)
            pred_distri: (B, N, 4 * reg_max)
        """
        from detect_fixed import DFL
        
        batch_size = cls_preds[0].shape[0]
        device = cls_preds[0].device
        
        # Create DFL module for decoding
        dfl = DFL(self.reg_max * 4, reg_max=self.reg_max).to(device)
        
        all_pred_bboxes = []
        all_pred_scores = []
        all_pred_distri = []
        
        anchor_idx = 0
        for box_pred, cls_pred in zip(box_preds, cls_preds):
            b, _, h, w = box_pred.shape
            num_anchors = h * w
            
            # Get distances from DFL
            distances = dfl(box_pred)  # (B, 4, H, W)
            distances = distances.view(b, 4, -1).permute(0, 2, 1)  # (B, N, 4)
            
            # Get anchor points for this level
            anchors = anchor_points[anchor_idx:anchor_idx + num_anchors].unsqueeze(0)
            anchors = anchors.expand(b, -1, -1)
            
            # Decode to boxes
            from coordinate_utils import dist2bbox
            boxes = dist2bbox(distances, anchors, xywh=False)
            
            # Process class predictions
            scores = cls_pred.view(b, self.num_classes, -1).permute(0, 2, 1)
            
            # Store distribution for DFL loss
            distri = box_pred.view(b, 4, self.reg_max, -1).permute(0, 3, 1, 2).reshape(b, -1, 4 * self.reg_max)
            
            all_pred_bboxes.append(boxes)
            all_pred_scores.append(scores)
            all_pred_distri.append(distri)
            
            anchor_idx += num_anchors
        
        pred_bboxes = torch.cat(all_pred_bboxes, dim=1)
        pred_scores = torch.cat(all_pred_scores, dim=1)
        pred_distri = torch.cat(all_pred_distri, dim=1)
        
        return pred_bboxes, pred_scores, pred_distri
