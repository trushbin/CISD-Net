# CISD-Net: 3D YOLO Object Detection Framework

A complete, production-ready 3D YOLO object detection framework with proper coordinate handling, dynamic image size support, and comprehensive bug fixes.

## Features

### Core Improvements

1. **✅ Coordinate System Unification**
   - Uses absolute coordinates throughout (no normalization confusion)
   - Clear conversion functions when normalization is needed
   - Validation functions to ensure coordinates are within bounds

2. **✅ Fixed Stride Initialization**
   - Correct stride values: `[4, 8, 16]` for 3 detection levels (not `[8, 16]`)
   - Proper multi-scale detection

3. **✅ Dynamic Image Size Support**
   - No hardcoded dimensions
   - Supports any input size (e.g., 320x320, 640x640, 800x800)
   - Automatic anchor generation based on feature map sizes

4. **✅ Fixed Data Pipeline**
   - Consistent label format (absolute coordinates internally)
   - Proper data augmentation with coordinate transformations
   - Validation at each step

5. **✅ Fixed DFL Module**
   - Correct projection buffer dimensions
   - Proper softmax over distribution dimension

6. **✅ Task-Aligned Assigner**
   - Dynamic image size support
   - Better positive/negative sample selection
   - Task-aligned metric (combines cls and IoU)

7. **✅ Coordinate Validity**
   - Validation at each step
   - Clipping to image bounds
   - Box format conversions (xyxy ↔ xywh)

8. **✅ Gradient Flow**
   - All tensors maintain `requires_grad` properly
   - Mixed precision training support
   - Gradient clipping

9. **✅ Model Export/Import**
   - Consistent state_dict handling
   - ONNX export support
   - Easy checkpoint resuming

10. **✅ Comprehensive Validation**
    - Unit tests for all coordinate operations
    - End-to-end model testing
    - Multiple image size validation

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
CISD-Net/
├── coordinate_utils.py          # Coordinate transformation utilities
├── detect_fixed.py               # Fixed Detect module with correct stride
├── loss_fixed.py                 # Fixed loss computation
├── task_fixed.py                 # Complete YOLO3D model
├── loader_fixed.py               # Fixed data loading
├── train_fixed.py                # Training script
├── val_fixed.py                  # Validation script
├── test_coordinate_system.py     # Validation tests
├── requirements.txt              # Python dependencies
└── final_NWPU/best.pt           # Pre-trained weights (NWPU dataset)
└── final_SIMD/best.pt           # Pre-trained weights (SIMD dataset)
```

## Usage

### 1. Test Coordinate System

Validate that all coordinate transformations work correctly:

```bash
python test_coordinate_system.py
```

Expected output:
```
============================================================
Running Coordinate System Validation Tests
============================================================

=== Testing Coordinate Conversions ===
✓ xyxy2xywh conversion correct
✓ xywh2xyxy conversion correct
...

============================================================
✓ All tests passed!
============================================================
```

### 2. Training

Train a model from scratch or fine-tune pre-trained weights:

```bash
python train_fixed.py \
    --train-img-dir /path/to/train/images \
    --train-label-dir /path/to/train/labels \
    --val-img-dir /path/to/val/images \
    --val-label-dir /path/to/val/labels \
    --num-classes 80 \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --save-dir ./runs/train
```

**Key Arguments:**
- `--train-img-dir`: Directory containing training images
- `--train-label-dir`: Directory containing training labels (YOLO format)
- `--val-img-dir`: Directory containing validation images (optional)
- `--val-label-dir`: Directory containing validation labels (optional)
- `--num-classes`: Number of object classes
- `--pretrained`: Path to pre-trained weights (optional)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--img-size`: Input image size (e.g., 640)
- `--amp`: Enable automatic mixed precision training
- `--resume`: Resume from checkpoint

**Label Format:**
YOLO format - one `.txt` file per image with lines:
```
class_id x_center y_center width height
```
where coordinates are normalized to [0, 1].

### 3. Validation

Validate model performance:

```bash
python val_fixed.py \
    --weights ./runs/train/best.pt \
    --img-dir /path/to/val/images \
    --label-dir /path/to/val/labels \
    --img-size 640 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --save-dir ./runs/val
```

### 4. Inference

Run inference on images without labels:

```bash
python val_fixed.py \
    --weights ./runs/train/best.pt \
    --img-dir /path/to/images \
    --img-size 640 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --save-dir ./runs/inference \
    --visualize
```

Results will be saved to `--save-dir`:
- Detection text files (one per image)
- Visualizations (if `--visualize` is used)

### 5. Using Pre-trained Weights

Load the provided NWPU or SIMD pre-trained weights:

```python
from task_fixed import YOLO3DModel

# Load model
model, checkpoint = YOLO3DModel.load('final_NWPU/best.pt', device='cuda')
model.eval()

# Inference
import torch
img = torch.randn(1, 3, 640, 640).cuda()
predictions = model(img)
```

## API Reference

### Coordinate Utils

```python
from coordinate_utils import (
    xyxy2xywh,           # Convert [x1,y1,x2,y2] to [cx,cy,w,h]
    xywh2xyxy,           # Convert [cx,cy,w,h] to [x1,y1,x2,y2]
    normalize_coordinates,    # Normalize to [0, 1]
    denormalize_coordinates,  # Denormalize to pixels
    clip_boxes,          # Clip boxes to image bounds
    validate_boxes,      # Check if boxes are valid
    box_iou,            # Compute IoU between boxes
    dist2bbox,          # Distance predictions to boxes
    bbox2dist,          # Boxes to distance predictions
    make_anchors        # Generate anchor points
)
```

### Model

```python
from task_fixed import YOLO3DModel, create_model

# Create new model
model = create_model(num_classes=80, width_mult=1.0)

# Load pre-trained model
model, checkpoint = YOLO3DModel.load('path/to/weights.pt')

# Save model
model.save('path/to/save.pt', optimizer=optimizer, epoch=10)

# Export to ONNX
model.export_onnx('model.onnx', input_shape=(1, 3, 640, 640))
```

### Data Loading

```python
from loader_fixed import create_dataloader

# Create training dataloader
train_loader = create_dataloader(
    img_dir='path/to/images',
    label_dir='path/to/labels',
    batch_size=16,
    img_size=640,
    augment=True,
    num_workers=4
)
```

## Technical Details

### Coordinate System

The framework uses **absolute coordinates (pixels)** throughout:

1. **Loading**: Labels are loaded in YOLO format (normalized) and immediately converted to absolute coordinates
2. **Processing**: All internal operations use absolute coordinates
3. **Loss**: Coordinates are used directly in absolute form
4. **Output**: Predictions are in absolute coordinates

This eliminates confusion and errors from mixing normalized and absolute coordinates.

### Stride Configuration

The detection head uses three scales with strides:
- **P3**: stride = 4 (high resolution for small objects)
- **P4**: stride = 8 (medium resolution)
- **P5**: stride = 16 (low resolution for large objects)

This is different from typical YOLO which uses [8, 16, 32]. Using smaller strides improves small object detection.

### DFL (Distribution Focal Loss)

The DFL module converts distribution predictions to box distances:
- Input: `(B, reg_max * 4, H, W)` where `reg_max=16`
- Process: Reshape to `(B, 4, reg_max, H, W)`, apply softmax, project
- Output: `(B, 4, H, W)` - expected distance values

### Task-Aligned Assigner

Assigns ground truth to predictions using:
```
alignment_metric = score^alpha * IoU^beta
```
where `alpha=1.0` and `beta=6.0`. This ensures both classification and localization quality are considered.

## Testing

Run the comprehensive test suite:

```bash
python test_coordinate_system.py
```

Tests include:
- Coordinate format conversions
- Box clipping and validation
- IoU computation
- Distance ↔ bbox conversions
- Anchor generation
- DFL module
- Detect module with correct strides
- Gradient flow
- Dynamic image sizes
- State dict handling

All tests should pass with ✓ markers.

## Performance Tips

1. **Mixed Precision Training**: Use `--amp` flag for faster training
2. **Batch Size**: Adjust based on GPU memory (16-32 recommended)
3. **Image Size**: Start with 640, increase for better accuracy
4. **Learning Rate**: 0.01 for SGD, 0.001 for Adam/AdamW
5. **Data Augmentation**: Enabled by default for training
6. **Gradient Clipping**: Default threshold of 10.0 prevents explosion

## Common Issues

### Issue: Model outputs NaN loss

**Solution**: Check that:
- Ground truth boxes are valid (within image bounds)
- Learning rate is not too high
- Batch size is appropriate

### Issue: Low recall

**Solution**:
- Decrease `--conf-thres` during inference
- Increase training epochs
- Check data augmentation is not too aggressive

### Issue: Slow training

**Solution**:
- Enable mixed precision with `--amp`
- Reduce `--num-workers` if CPU-bound
- Use smaller `--img-size`

## Citation

If you use this code, please cite:

```bibtex
@misc{cisd-net-2025,
  title={CISD-Net: 3D YOLO Object Detection Framework},
  author={CISD-Net Contributors},
  year={2025},
  howpublished={\url{https://github.com/trushbin/CISD-Net}}
}
```

## License

This project is open-sourced under standard research and educational use terms.

## Acknowledgments

This framework fixes and extends standard YOLO architectures with:
- Proper coordinate handling
- Dynamic image size support  
- Improved detection head design
- Comprehensive validation

Pre-trained weights are provided for NWPU and SIMD datasets.
