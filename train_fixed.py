"""
Fixed Training Script for 3D YOLO

Key features:
- Proper gradient flow maintenance
- Dynamic image size support
- Coordinate validation throughout
- Comprehensive logging
"""

import torch
import torch.optim as optim
from torch.cuda import amp
import argparse
from pathlib import Path
import time
from tqdm import tqdm

from task_fixed import YOLO3DModel, create_model
from loader_fixed import create_dataloader
from loss_fixed import ComputeLoss


class Trainer:
    """Training manager for YOLO3D."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        print(f"Creating model with {args.num_classes} classes...")
        self.model = create_model(
            num_classes=args.num_classes,
            width_mult=args.width_mult,
            pretrained_path=args.pretrained if args.pretrained else None,
            device=self.device
        )
        
        # Create dataloaders
        print(f"Loading training data from {args.train_img_dir}...")
        self.train_loader = create_dataloader(
            img_dir=args.train_img_dir,
            label_dir=args.train_label_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            augment=True,
            num_workers=args.num_workers,
            shuffle=True
        )
        
        if args.val_img_dir:
            print(f"Loading validation data from {args.val_img_dir}...")
            self.val_loader = create_dataloader(
                img_dir=args.val_img_dir,
                label_dir=args.val_label_dir,
                batch_size=args.batch_size,
                img_size=args.img_size,
                augment=False,
                num_workers=args.num_workers,
                shuffle=False
            )
        else:
            self.val_loader = None
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = amp.GradScaler(enabled=args.amp)
        
        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # Load checkpoint if resuming
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _create_optimizer(self):
        """Create optimizer with proper parameter groups."""
        # Separate parameters for different learning rates
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, torch.nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay for BN weights
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
                pg1.append(v.weight)  # apply decay
        
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(pg0, lr=self.args.lr, betas=(0.9, 0.999))
        elif self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(pg0, lr=self.args.lr, betas=(0.9, 0.999))
        else:  # SGD
            optimizer = optim.SGD(pg0, lr=self.args.lr, momentum=0.937, nesterov=True)
        
        optimizer.add_param_group({'params': pg1, 'weight_decay': self.args.weight_decay})
        optimizer.add_param_group({'params': pg2})
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.epochs // 3, gamma=0.1
            )
        else:
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda x: 1.0)
        
        return scheduler
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        
        total_loss = 0
        total_loss_cls = 0
        total_loss_iou = 0
        total_loss_dfl = 0
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass with automatic mixed precision
            with amp.autocast(enabled=self.args.amp):
                loss, loss_dict = self.model(images, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss_dict['loss']
            total_loss_cls += loss_dict['loss_cls']
            total_loss_iou += loss_dict['loss_iou']
            total_loss_dfl += loss_dict['loss_dfl']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'cls': f"{loss_dict['loss_cls']:.4f}",
                'iou': f"{loss_dict['loss_iou']:.4f}",
                'dfl': f"{loss_dict['loss_dfl']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate average losses
        num_batches = len(self.train_loader)
        avg_metrics = {
            'loss': total_loss / num_batches,
            'loss_cls': total_loss_cls / num_batches,
            'loss_iou': total_loss_iou / num_batches,
            'loss_dfl': total_loss_dfl / num_batches
        }
        
        return avg_metrics
    
    def validate(self, epoch):
        """Validate the model."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        total_loss = 0
        total_loss_cls = 0
        total_loss_iou = 0
        total_loss_dfl = 0
        
        pbar = tqdm(self.val_loader, desc=f"Validation")
        
        with torch.no_grad():
            for images, targets in pbar:
                # Move to device
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                loss, loss_dict = self.model(images, targets)
                
                # Update metrics
                total_loss += loss_dict['loss']
                total_loss_cls += loss_dict['loss_cls']
                total_loss_iou += loss_dict['loss_iou']
                total_loss_dfl += loss_dict['loss_dfl']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss']:.4f}"
                })
        
        # Calculate average losses
        num_batches = len(self.val_loader)
        avg_metrics = {
            'loss': total_loss / num_batches,
            'loss_cls': total_loss_cls / num_batches,
            'loss_iou': total_loss_iou / num_batches,
            'loss_dfl': total_loss_dfl / num_batches
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        
        self.model.save(
            checkpoint_path,
            optimizer=self.optimizer,
            epoch=epoch,
            scheduler=self.scheduler,
            best_loss=self.best_loss
        )
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best.pt'
            self.model.save(best_path, epoch=epoch, best_loss=self.best_loss)
            print(f"Saved best model to {best_path}")
        
        # Keep only last N checkpoints
        if self.args.keep_checkpoints > 0:
            checkpoints = sorted(self.save_dir.glob('checkpoint_epoch_*.pt'))
            if len(checkpoints) > self.args.keep_checkpoints:
                for ckpt in checkpoints[:-self.args.keep_checkpoints]:
                    ckpt.unlink()
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Image size: {self.args.img_size}")
        print(f"Number of classes: {self.args.num_classes}")
        print(f"Save directory: {self.save_dir}\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch) if self.val_loader else None
            
            # Update scheduler
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(cls: {train_metrics['loss_cls']:.4f}, "
                  f"iou: {train_metrics['loss_iou']:.4f}, "
                  f"dfl: {train_metrics['loss_dfl']:.4f})")
            
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f} "
                      f"(cls: {val_metrics['loss_cls']:.4f}, "
                      f"iou: {val_metrics['loss_iou']:.4f}, "
                      f"dfl: {val_metrics['loss_dfl']:.4f})")
            
            # Save checkpoint
            current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
            is_best = current_loss < self.best_loss
            
            if is_best:
                self.best_loss = current_loss
            
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        print(f"\nTraining completed! Best loss: {self.best_loss:.4f}")
        print(f"Models saved to {self.save_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO3D Model')
    
    # Data
    parser.add_argument('--train-img-dir', type=str, required=True,
                       help='Training images directory')
    parser.add_argument('--train-label-dir', type=str, required=True,
                       help='Training labels directory')
    parser.add_argument('--val-img-dir', type=str, default=None,
                       help='Validation images directory')
    parser.add_argument('--val-label-dir', type=str, default=None,
                       help='Validation labels directory')
    
    # Model
    parser.add_argument('--num-classes', type=int, default=80,
                       help='Number of classes')
    parser.add_argument('--width-mult', type=float, default=1.0,
                       help='Model width multiplier')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='SGD',
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                       help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Checkpoints
    parser.add_argument('--save-dir', type=str, default='./runs/train',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--keep-checkpoints', type=int, default=5,
                       help='Number of checkpoints to keep (0 for all)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
