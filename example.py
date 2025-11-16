"""
Example script demonstrating CISD-Net 3D YOLO framework usage.

This script shows:
1. Loading a pre-trained model
2. Running inference on a test image
3. Visualizing results
"""

import torch
import numpy as np
import cv2
from pathlib import Path

# Check if dependencies are installed
try:
    from task_fixed import YOLO3DModel
    from val_fixed import non_max_suppression, scale_boxes
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    exit(1)


def create_demo_image():
    """Create a simple test image with random data."""
    # Create a synthetic image (640x640x3)
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Add some simple shapes for visual interest
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 3)
    cv2.circle(img, (400, 200), 50, (255, 0, 0), 3)
    cv2.putText(img, "DEMO IMAGE", (200, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def demo_model_loading():
    """Demonstrate loading a pre-trained model."""
    print("\n" + "="*60)
    print("Demo 1: Model Loading")
    print("="*60)
    
    # Check if pre-trained weights exist
    weight_paths = [
        'final_NWPU/best.pt',
        'final_SIMD/best.pt'
    ]
    
    weights_found = None
    for path in weight_paths:
        if Path(path).exists():
            weights_found = path
            break
    
    if weights_found:
        print(f"✓ Found pre-trained weights: {weights_found}")
        
        # Load model
        print("Loading model...")
        model, checkpoint = YOLO3DModel.load(weights_found, device='cpu')
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  - Number of classes: {model.num_classes}")
        print(f"  - Device: {next(model.parameters()).device}")
        
        if 'epoch' in checkpoint:
            print(f"  - Trained for: {checkpoint['epoch']} epochs")
        
        return model
    else:
        print("✗ No pre-trained weights found")
        print("Creating a new model for demonstration...")
        
        model = YOLO3DModel(num_classes=10)
        model.eval()
        
        print(f"✓ New model created")
        print(f"  - Number of classes: {model.num_classes}")
        
        return model


def demo_inference(model):
    """Demonstrate running inference."""
    print("\n" + "="*60)
    print("Demo 2: Inference")
    print("="*60)
    
    # Create demo image
    print("Creating demo image...")
    img = create_demo_image()
    cv2.imwrite('demo_input.jpg', img)
    print("✓ Demo image saved as 'demo_input.jpg'")
    
    # Prepare image for model
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    print(f"Image tensor shape: {img_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(img_tensor)
    
    print(f"✓ Inference complete")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Batch size: {predictions.shape[0]}")
    print(f"  - Number of anchors: {predictions.shape[1]}")
    print(f"  - Channels (4 bbox + classes): {predictions.shape[2]}")
    
    return predictions, img


def demo_nms(predictions, model):
    """Demonstrate Non-Maximum Suppression."""
    print("\n" + "="*60)
    print("Demo 3: Non-Maximum Suppression")
    print("="*60)
    
    # Apply NMS
    conf_thres = 0.25
    iou_thres = 0.45
    
    print(f"Applying NMS (conf_thres={conf_thres}, iou_thres={iou_thres})...")
    detections = non_max_suppression(
        predictions,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=model.num_classes
    )[0]
    
    print(f"✓ NMS complete")
    print(f"  - Number of detections: {len(detections)}")
    
    if len(detections) > 0:
        print(f"  - Detection format: [x1, y1, x2, y2, conf, class]")
        print(f"\n  Top 3 detections:")
        for i, det in enumerate(detections[:3]):
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            print(f"    {i+1}. Class {int(cls)}, Conf: {conf:.3f}, "
                  f"Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    else:
        print("  - No detections above confidence threshold")
        print("  - Note: This is expected for random demo images")
    
    return detections


def demo_visualization(img, detections, model):
    """Demonstrate visualization of detections."""
    print("\n" + "="*60)
    print("Demo 4: Visualization")
    print("="*60)
    
    if len(detections) == 0:
        print("No detections to visualize (expected for demo image)")
        
        # Create a visualization anyway showing the framework works
        img_vis = img.copy()
        cv2.putText(img_vis, "No detections (demo image)", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_vis, "Framework is working correctly!", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite('demo_output.jpg', img_vis)
        print("✓ Saved demo output to 'demo_output.jpg'")
        return
    
    # Draw detections
    img_vis = img.copy()
    
    print("Drawing detections...")
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        
        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        label = f"Class {int(cls)}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Background for label
        cv2.rectangle(img_vis, 
                     (int(x1), int(y1) - label_size[1] - 10),
                     (int(x1) + label_size[0], int(y1)),
                     color, -1)
        
        # Label text
        cv2.putText(img_vis, label, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save result
    cv2.imwrite('demo_output.jpg', img_vis)
    print(f"✓ Saved visualization to 'demo_output.jpg'")


def demo_dynamic_image_sizes():
    """Demonstrate dynamic image size support."""
    print("\n" + "="*60)
    print("Demo 5: Dynamic Image Sizes")
    print("="*60)
    
    model = YOLO3DModel(num_classes=10)
    model.eval()
    
    test_sizes = [(320, 320), (640, 640), (800, 800)]
    
    print("Testing different input sizes...")
    for h, w in test_sizes:
        img = torch.randn(1, 3, h, w)
        
        with torch.no_grad():
            output = model(img)
        
        print(f"✓ Size {(h, w)}: Output shape {output.shape}, "
              f"Anchors: {output.shape[1]}")
    
    print("\n✓ Model supports dynamic image sizes!")


def main():
    """Run all demonstrations."""
    print("="*60)
    print("CISD-Net 3D YOLO Framework - Example Script")
    print("="*60)
    
    try:
        # Demo 1: Load model
        model = demo_model_loading()
        
        # Demo 2: Run inference
        predictions, img = demo_inference(model)
        
        # Demo 3: Apply NMS
        detections = demo_nms(predictions, model)
        
        # Demo 4: Visualize results
        demo_visualization(img, detections, model)
        
        # Demo 5: Dynamic image sizes
        demo_dynamic_image_sizes()
        
        print("\n" + "="*60)
        print("✓ All demos completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  - demo_input.jpg: Input image")
        print("  - demo_output.jpg: Output with detections (if any)")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for training on your data")
        print("  2. Read README_FRAMEWORK.md for detailed documentation")
        print("  3. Run test_coordinate_system.py for validation")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
