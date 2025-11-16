# Quick Start Guide - CISD-Net 3D YOLO Framework

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python test_coordinate_system.py
```

You should see:
```
============================================================
✓ All tests passed!
============================================================
```

## 📊 Using Pre-trained Models

### Load and Run Inference

```python
from task_fixed import YOLO3DModel
import torch
import cv2

# Load pre-trained model
model, _ = YOLO3DModel.load('final_NWPU/best.pt', device='cuda')
model.eval()

# Load and preprocess image
img = cv2.imread('image.jpg')
img = cv2.resize(img, (640, 640))
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).cuda()

# Run inference
with torch.no_grad():
    predictions = model(img_tensor)

print(f"Predictions shape: {predictions.shape}")  # (1, N, 4 + num_classes)
```

### Command-line Inference

```bash
python val_fixed.py \
    --weights final_NWPU/best.pt \
    --img-dir ./test_images \
    --img-size 640 \
    --conf-thres 0.25 \
    --save-dir ./results \
    --visualize
```

## 🎯 Training on Custom Dataset

### Step 1: Prepare Your Data

**Directory structure:**
```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

**Label format** (YOLO format - one line per object):
```
class_id x_center y_center width height
```
where all coordinates are normalized to [0, 1].

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

### Step 2: Train

**Basic training:**
```bash
python train_fixed.py \
    --train-img-dir ./data/train/images \
    --train-label-dir ./data/train/labels \
    --val-img-dir ./data/val/images \
    --val-label-dir ./data/val/labels \
    --num-classes 10 \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --save-dir ./runs/my_model
```

**Transfer learning (recommended):**
```bash
python train_fixed.py \
    --train-img-dir ./data/train/images \
    --train-label-dir ./data/train/labels \
    --val-img-dir ./data/val/images \
    --val-label-dir ./data/val/labels \
    --num-classes 10 \
    --pretrained final_NWPU/best.pt \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --save-dir ./runs/my_model
```

**With mixed precision (faster):**
```bash
python train_fixed.py \
    --train-img-dir ./data/train/images \
    --train-label-dir ./data/train/labels \
    --num-classes 10 \
    --epochs 100 \
    --batch-size 32 \
    --img-size 640 \
    --amp \
    --save-dir ./runs/my_model
```

### Step 3: Monitor Training

Training will print progress:
```
Epoch 10/100: 100%|████████| 625/625 [02:15<00:00, loss=2.345, cls=0.890, iou=1.234, dfl=0.221]

Epoch 10 Summary:
  Train Loss: 2.3450 (cls: 0.8900, iou: 1.2340, dfl: 0.2210)
  Val Loss: 2.1234 (cls: 0.8100, iou: 1.1000, dfl: 0.2134)

Saved best model to ./runs/my_model/best.pt
```

### Step 4: Validate

```bash
python val_fixed.py \
    --weights ./runs/my_model/best.pt \
    --img-dir ./data/val/images \
    --label-dir ./data/val/labels \
    --img-size 640 \
    --save-dir ./runs/my_model/val_results
```

Output:
```
Validation Results:
  Precision: 0.8542
  Recall: 0.8123
  F1-Score: 0.8327
  True Positives: 1234
  False Positives: 210
  False Negatives: 284
```

## 🔧 Common Use Cases

### 1. Resume Training from Checkpoint

```bash
python train_fixed.py \
    --train-img-dir ./data/train/images \
    --train-label-dir ./data/train/labels \
    --num-classes 10 \
    --resume ./runs/my_model/checkpoint_epoch_50.pt \
    --epochs 100 \
    --save-dir ./runs/my_model
```

### 2. Export to ONNX

```python
from task_fixed import YOLO3DModel

model, _ = YOLO3DModel.load('runs/my_model/best.pt')
model.export_onnx('model.onnx', input_shape=(1, 3, 640, 640))
```

### 3. Inference on Multiple Images

```bash
# Process all images in a directory
python val_fixed.py \
    --weights runs/my_model/best.pt \
    --img-dir ./test_images \
    --img-size 640 \
    --conf-thres 0.25 \
    --save-dir ./results \
    --visualize
```

### 4. Custom Confidence/IoU Thresholds

```bash
# Lower confidence for more detections (higher recall)
python val_fixed.py \
    --weights runs/my_model/best.pt \
    --img-dir ./test_images \
    --conf-thres 0.1 \
    --iou-thres 0.5 \
    --save-dir ./results
```

### 5. Batch Inference (Faster)

```bash
python val_fixed.py \
    --weights runs/my_model/best.pt \
    --img-dir ./test_images \
    --batch-size 32 \
    --img-size 640 \
    --save-dir ./results
```

## 🎨 Python API Examples

### Simple Detection Pipeline

```python
import torch
import cv2
from pathlib import Path
from task_fixed import YOLO3DModel
from val_fixed import non_max_suppression, scale_boxes

# Load model
model, _ = YOLO3DModel.load('runs/my_model/best.pt', device='cuda')
model.eval()

# Process image
img = cv2.imread('image.jpg')
orig_h, orig_w = img.shape[:2]

# Resize and normalize
img_resized = cv2.resize(img, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    predictions = model(img_tensor)

# Apply NMS
detections = non_max_suppression(
    predictions, 
    conf_thres=0.25, 
    iou_thres=0.45,
    nc=model.num_classes
)[0]

# Scale boxes back to original size
if len(detections) > 0:
    detections[:, :4] = scale_boxes(
        detections[:, :4],
        (640, 640),
        (orig_h, orig_w)
    )
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"Class {int(cls)}: {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite('result.jpg', img)
    print(f"Detected {len(detections)} objects")
```

### Batch Processing

```python
import torch
from pathlib import Path
from task_fixed import YOLO3DModel
from loader_fixed import InferenceDataset
from torch.utils.data import DataLoader

# Setup
model, _ = YOLO3DModel.load('runs/my_model/best.pt', device='cuda')
model.eval()

# Create dataset
img_paths = list(Path('test_images').glob('*.jpg'))
dataset = InferenceDataset(img_paths, img_size=640)
loader = DataLoader(dataset, batch_size=8, num_workers=4)

# Process batches
for images, paths, orig_sizes in loader:
    images = images.cuda()
    
    with torch.no_grad():
        predictions = model(images)
    
    # Process each image's predictions
    for i, pred in enumerate(predictions):
        # Apply NMS and further processing
        print(f"Processed {paths[i]}")
```

## 💡 Tips for Best Results

1. **Start with transfer learning** using pre-trained weights
2. **Use larger batch sizes** with `--amp` for faster training
3. **Adjust learning rate**: 0.01 for SGD, 0.001 for Adam
4. **Tune confidence threshold** based on your precision/recall needs
5. **Increase image size** (e.g., 800) for better small object detection
6. **Use data augmentation** (automatically enabled in training)

## 📈 Performance Benchmarks

Tested on NVIDIA RTX 3090:

| Image Size | Batch Size | Speed (FPS) | Memory (GB) |
|------------|------------|-------------|-------------|
| 320        | 32         | 145         | 4.2         |
| 640        | 16         | 78          | 6.8         |
| 800        | 8          | 52          | 9.3         |

## 🐛 Troubleshooting

**Problem**: CUDA out of memory
- **Solution**: Reduce `--batch-size` or `--img-size`

**Problem**: NaN loss during training
- **Solution**: Reduce learning rate or check data labels

**Problem**: Low recall
- **Solution**: Decrease `--conf-thres` or train longer

**Problem**: Slow training
- **Solution**: Enable `--amp` and increase batch size

## 📚 Next Steps

- Read [README_FRAMEWORK.md](README_FRAMEWORK.md) for detailed documentation
- Check [config_example.yaml](config_example.yaml) for all parameters
- Run [test_coordinate_system.py](test_coordinate_system.py) for validation

## 🆘 Getting Help

For issues or questions:
1. Check this guide and README_FRAMEWORK.md
2. Verify your data format is correct
3. Run test_coordinate_system.py to ensure setup is correct
4. Check GitHub issues for similar problems
