"""
Test suite for coordinate system validation.

Validates that:
1. Coordinate transformations are correct
2. Boxes remain within image bounds
3. All operations maintain absolute coordinates
4. Gradient flow is maintained
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from coordinate_utils import (
    xyxy2xywh, xywh2xyxy, normalize_coordinates, denormalize_coordinates,
    clip_boxes, validate_boxes, box_area, box_iou, dist2bbox, bbox2dist,
    make_anchors
)
from detect_fixed import Detect, DFL
from task_fixed import YOLO3DModel


def test_coordinate_conversions():
    """Test coordinate format conversions."""
    print("\n=== Testing Coordinate Conversions ===")
    
    # Test xyxy to xywh
    boxes_xyxy = torch.tensor([
        [10.0, 20.0, 50.0, 80.0],
        [0.0, 0.0, 100.0, 100.0]
    ])
    
    boxes_xywh = xyxy2xywh(boxes_xyxy)
    expected_xywh = torch.tensor([
        [30.0, 50.0, 40.0, 60.0],  # center=(30, 50), size=(40, 60)
        [50.0, 50.0, 100.0, 100.0]
    ])
    
    assert torch.allclose(boxes_xywh, expected_xywh), "xyxy2xywh conversion failed"
    print("✓ xyxy2xywh conversion correct")
    
    # Test xywh to xyxy
    boxes_xyxy_back = xywh2xyxy(boxes_xywh)
    assert torch.allclose(boxes_xyxy_back, boxes_xyxy), "xywh2xyxy conversion failed"
    print("✓ xywh2xyxy conversion correct")
    
    # Test normalization
    img_w, img_h = 640, 480
    boxes_norm = normalize_coordinates(boxes_xyxy, img_w, img_h)
    expected_norm = torch.tensor([
        [10.0/640, 20.0/480, 50.0/640, 80.0/480],
        [0.0, 0.0, 100.0/640, 100.0/480]
    ])
    assert torch.allclose(boxes_norm, expected_norm), "Normalization failed"
    print("✓ Coordinate normalization correct")
    
    # Test denormalization
    boxes_denorm = denormalize_coordinates(boxes_norm, img_w, img_h)
    assert torch.allclose(boxes_denorm, boxes_xyxy), "Denormalization failed"
    print("✓ Coordinate denormalization correct")


def test_box_clipping():
    """Test box clipping to image boundaries."""
    print("\n=== Testing Box Clipping ===")
    
    img_w, img_h = 640, 480
    
    # Boxes that extend outside image
    boxes_xyxy = torch.tensor([
        [-10.0, -5.0, 650.0, 490.0],
        [10.0, 20.0, 50.0, 80.0]
    ])
    
    clipped = clip_boxes(boxes_xyxy, img_w, img_h, format='xyxy')
    
    assert (clipped[:, 0] >= 0).all(), "x1 not clipped to 0"
    assert (clipped[:, 1] >= 0).all(), "y1 not clipped to 0"
    assert (clipped[:, 2] <= img_w).all(), "x2 not clipped to img_w"
    assert (clipped[:, 3] <= img_h).all(), "y2 not clipped to img_h"
    
    print("✓ Box clipping correct")


def test_box_validation():
    """Test box validation."""
    print("\n=== Testing Box Validation ===")
    
    img_w, img_h = 640, 480
    
    # Mix of valid and invalid boxes
    boxes_xyxy = torch.tensor([
        [10.0, 20.0, 50.0, 80.0],    # Valid
        [-10.0, 20.0, 50.0, 80.0],   # Invalid - x1 < 0
        [10.0, 20.0, 700.0, 80.0],   # Invalid - x2 > img_w
        [50.0, 20.0, 40.0, 80.0],    # Invalid - x2 < x1
    ])
    
    valid = validate_boxes(boxes_xyxy, img_w, img_h, format='xyxy')
    expected = torch.tensor([True, False, False, False])
    
    assert torch.equal(valid, expected), f"Validation failed: {valid} vs {expected}"
    print("✓ Box validation correct")


def test_box_iou():
    """Test IoU computation."""
    print("\n=== Testing Box IoU ===")
    
    boxes1 = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 15.0, 15.0]
    ])
    
    boxes2 = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [10.0, 10.0, 20.0, 20.0]
    ])
    
    iou = box_iou(boxes1, boxes2, format='xyxy')
    
    # First box with itself should be 1.0
    assert torch.isclose(iou[0, 0], torch.tensor(1.0)), "Self IoU should be 1.0"
    
    # Non-overlapping boxes should have IoU = 0
    assert torch.isclose(iou[0, 1], torch.tensor(0.0)), "Non-overlapping IoU should be 0"
    
    print("✓ Box IoU computation correct")


def test_dist2bbox():
    """Test distance to bbox conversion."""
    print("\n=== Testing dist2bbox ===")
    
    # Anchor at (50, 50) with distances [10, 10, 10, 10]
    anchor_points = torch.tensor([[50.0, 50.0]])
    distances = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    
    # Expected: bbox from (40, 40) to (60, 60)
    boxes_xyxy = dist2bbox(distances, anchor_points, xywh=False)
    expected = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
    
    assert torch.allclose(boxes_xyxy, expected), f"dist2bbox failed: {boxes_xyxy} vs {expected}"
    print("✓ dist2bbox conversion correct")


def test_bbox2dist():
    """Test bbox to distance conversion."""
    print("\n=== Testing bbox2dist ===")
    
    # Box from (40, 40) to (60, 60) with anchor at (50, 50)
    boxes = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
    anchor_points = torch.tensor([[50.0, 50.0]])
    
    distances = bbox2dist(boxes, anchor_points)
    expected = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    
    assert torch.allclose(distances, expected), f"bbox2dist failed: {distances} vs {expected}"
    print("✓ bbox2dist conversion correct")


def test_make_anchors():
    """Test anchor generation."""
    print("\n=== Testing Anchor Generation ===")
    
    # Create dummy feature maps
    feat1 = torch.randn(1, 256, 80, 80)  # Stride 8
    feat2 = torch.randn(1, 512, 40, 40)  # Stride 16
    feat3 = torch.randn(1, 1024, 20, 20) # Stride 32
    
    feats = [feat1, feat2, feat3]
    strides = torch.tensor([8.0, 16.0, 32.0])
    
    anchor_points, stride_tensor = make_anchors(feats, strides)
    
    # Total number of anchors
    expected_num = 80*80 + 40*40 + 20*20
    assert anchor_points.shape[0] == expected_num, f"Wrong number of anchors: {anchor_points.shape[0]} vs {expected_num}"
    
    # Check stride tensor
    assert stride_tensor.shape[0] == expected_num, "Stride tensor size mismatch"
    
    # Check that anchors are in absolute coordinates
    # First anchor should be at (stride * 0.5, stride * 0.5)
    first_anchor = anchor_points[0]
    expected_first = torch.tensor([4.0, 4.0])  # stride=8, offset=0.5 -> 8*0.5=4
    assert torch.allclose(first_anchor, expected_first), f"First anchor wrong: {first_anchor} vs {expected_first}"
    
    print("✓ Anchor generation correct")


def test_dfl_module():
    """Test DFL module with correct dimensions."""
    print("\n=== Testing DFL Module ===")
    
    reg_max = 16
    dfl = DFL(c1=64, reg_max=reg_max)
    
    # Input: (B, reg_max * 4, H, W)
    x = torch.randn(2, 64, 40, 40)
    
    # Output should be (B, 4, H, W)
    output = dfl(x)
    
    assert output.shape == (2, 4, 40, 40), f"DFL output shape wrong: {output.shape}"
    
    # Check that proj buffer has correct shape
    assert dfl.proj.shape == (reg_max,), f"DFL proj buffer wrong shape: {dfl.proj.shape}"
    
    print("✓ DFL module correct")


def test_detect_module():
    """Test Detect module with correct strides."""
    print("\n=== Testing Detect Module ===")
    
    nc = 80
    ch = (256, 512, 1024)
    detect = Detect(nc=nc, ch=ch)
    
    # Check stride initialization
    expected_strides = torch.tensor([4.0, 8.0, 16.0])
    assert torch.equal(detect.stride, expected_strides), f"Stride wrong: {detect.stride} vs {expected_strides}"
    print("✓ Detect stride initialization correct [4, 8, 16]")
    
    # Test forward pass
    feat1 = torch.randn(2, 256, 80, 80)
    feat2 = torch.randn(2, 512, 40, 40)
    feat3 = torch.randn(2, 1024, 20, 20)
    
    feats = [feat1, feat2, feat3]
    
    # Training mode
    detect.train()
    box_preds, cls_preds, feat_out = detect(feats)
    
    assert len(box_preds) == 3, "Should have 3 box prediction levels"
    assert len(cls_preds) == 3, "Should have 3 class prediction levels"
    
    # Check shapes
    assert box_preds[0].shape == (2, 64, 80, 80), f"Box pred 0 shape wrong: {box_preds[0].shape}"
    assert cls_preds[0].shape == (2, nc, 80, 80), f"Cls pred 0 shape wrong: {cls_preds[0].shape}"
    
    print("✓ Detect module forward pass correct")
    
    # Inference mode
    detect.eval()
    with torch.no_grad():
        preds = detect(feats)
    
    # Should return (B, N, 4 + nc)
    assert preds.shape[0] == 2, "Batch size wrong"
    assert preds.shape[2] == 4 + nc, f"Prediction channels wrong: {preds.shape[2]} vs {4 + nc}"
    
    print("✓ Detect inference mode correct")


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("\n=== Testing Gradient Flow ===")
    
    # Create small model
    model = YOLO3DModel(num_classes=10)
    model.train()
    
    # Create dummy input
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    
    # Create dummy targets
    targets = {
        'labels': torch.randint(0, 10, (2, 5, 1)),
        'bboxes': torch.rand(2, 5, 4) * 100 + 10,  # Random boxes
        'mask': torch.tensor([[True, True, False, False, False],
                             [True, False, False, False, False]])
    }
    
    # Forward pass
    loss, loss_dict = model(x, targets)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "Input gradient is None"
    
    # Check that model parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "No model parameters have gradients"
    
    print("✓ Gradient flow maintained")


def test_dynamic_image_sizes():
    """Test that model works with different image sizes."""
    print("\n=== Testing Dynamic Image Sizes ===")
    
    model = YOLO3DModel(num_classes=10)
    model.eval()
    
    sizes = [(320, 320), (416, 416), (640, 640), (800, 800)]
    
    for h, w in sizes:
        x = torch.randn(1, 3, h, w)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape[0] == 1, f"Batch size wrong for size {(h, w)}"
        assert output.shape[2] == 14, f"Output channels wrong for size {(h, w)}"  # 4 + 10
        
        print(f"✓ Model works with size {(h, w)}, output shape: {output.shape}")


def test_state_dict_handling():
    """Test model save/load with consistent state dict."""
    print("\n=== Testing State Dict Handling ===")
    
    import tempfile
    import os
    
    # Create model
    model1 = YOLO3DModel(num_classes=20)
    
    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp_path = tmp.name
        model1.save(tmp_path, epoch=10)
    
    try:
        # Load model
        model2, checkpoint = YOLO3DModel.load(tmp_path)
        
        # Check that checkpoint has expected keys
        assert 'model' in checkpoint, "Checkpoint missing 'model' key"
        assert 'num_classes' in checkpoint, "Checkpoint missing 'num_classes' key"
        assert 'epoch' in checkpoint, "Checkpoint missing 'epoch' key"
        assert checkpoint['epoch'] == 10, "Epoch not saved correctly"
        
        # Check that models have same state
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert n1 == n2, f"Parameter names don't match: {n1} vs {n2}"
            assert torch.equal(p1, p2), f"Parameter values don't match for {n1}"
        
        print("✓ State dict save/load correct")
    finally:
        os.unlink(tmp_path)


def run_all_tests():
    """Run all coordinate system tests."""
    print("="*60)
    print("Running Coordinate System Validation Tests")
    print("="*60)
    
    try:
        test_coordinate_conversions()
        test_box_clipping()
        test_box_validation()
        test_box_iou()
        test_dist2bbox()
        test_bbox2dist()
        test_make_anchors()
        test_dfl_module()
        test_detect_module()
        test_gradient_flow()
        test_dynamic_image_sizes()
        test_state_dict_handling()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Test failed with error:")
        print(str(e))
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
