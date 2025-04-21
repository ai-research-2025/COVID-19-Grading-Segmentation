import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import load_model
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="COVID-19 CT Scan Segmentation Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained segmentation model (.keras or .h5)")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument("--output", type=str, required=True, help="Directory to save output predictions")
    parser.add_argument("--masks", type=str, default=None, help="Optional path to ground truth masks for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation (default: 0.5)")
    parser.add_argument("--img-size", type=int, default=128, help="Image size for inference (default: 128)")
    parser.add_argument("--save-overlay", action="store_true", help="Save overlay visualization of predictions on images")
    return parser.parse_args()

# Custom Dice Coefficient function
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Combined Dice and Binary Cross Entropy loss
def dice_bce_loss(y_true, y_pred):
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return 0.5 * dice_loss + 0.5 * bce_loss

# Function to preprocess image
def preprocess_image(image_path, img_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, img_size)
    image_normalized = image_resized / 255.0
    return image, image_normalized

# Function to preprocess mask
def preprocess_mask(mask_path, img_size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
        
    mask_resized = cv2.resize(mask, img_size)
    mask_normalized = mask_resized / 255.0
    return mask_resized, mask_normalized

# Prediction function
def predict_mask(model, image_normalized, img_size, threshold=0.5):
    # Add batch dimension
    input_tensor = np.expand_dims(image_normalized, axis=0)
    
    # Get prediction
    pred_mask = model.predict(input_tensor)[0]
    
    # Apply threshold
    pred_mask_binary = (pred_mask > threshold).astype(np.uint8)
    
    # Resize prediction to original size if needed
    pred_mask_resized = cv2.resize(pred_mask_binary.squeeze(), img_size)
    
    return pred_mask_binary, pred_mask_resized

# Function to create and save visualization
def save_visualization(original_image, pred_mask, true_mask=None, output_path=None, filename=None, show=False):
    if true_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        axes[2].imshow(pred_mask.squeeze(), cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(pred_mask.squeeze(), cmap='gray')
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")
    
    plt.tight_layout()
    
    if output_path and filename:
        plt.savefig(os.path.join(output_path, f"{filename}_visualization.png"), bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
        
# Function to create overlay visualization
def save_overlay(original_image, pred_mask, output_path, filename):
    # Create red overlay for segmentation
    overlay = original_image.copy()
    overlay[pred_mask.squeeze() > 0] = [255, 0, 0]  # Red color for segmented areas
    
    # Blend with original image
    alpha = 0.4
    blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(blended)
    plt.title("Segmentation Overlay")
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, f"{filename}_overlay.png"), bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
    
    # Load model
    print(f"Loading model from {args.model}")
    try:
        model = load_model(args.model, custom_objects={
            'dice_coefficient': dice_coefficient,
            'dice_bce_loss': dice_bce_loss
        })
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set image size
    img_size = (args.img_size, args.img_size)
    
    # Determine if input is a single image or directory
    if os.path.isdir(args.input):
        # Process directory of images
        image_paths = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        for root, _, files in os.walk(args.input):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(root, file))
        print(f"Found {len(image_paths)} images to process")
    else:
        # Process single image
        image_paths = [args.input]
        print(f"Processing single image: {args.input}")
    
    # Check if masks directory is provided for evaluation
    eval_mode = False
    mask_paths = {}
    if args.masks and os.path.isdir(args.masks):
        eval_mode = True
        print(f"Evaluation mode: Using masks from {args.masks}")
        for root, _, files in os.walk(args.masks):
            for file in files:
                if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']):
                    mask_paths[os.path.basename(file)] = os.path.join(root, file)
        print(f"Found {len(mask_paths)} mask files")
    
    # Create directory for binary masks
    masks_dir = os.path.join(args.output, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Create directory for visualizations
    viz_dir = os.path.join(args.output, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create directory for overlays if requested
    if args.save_overlay:
        overlay_dir = os.path.join(args.output, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)
    
    # Process images and collect results
    results = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.basename(image_path).split('.')[0]
        
        try:
            # Load and preprocess image
            original_image, image_normalized = preprocess_image(image_path, img_size)
            
            # Generate prediction
            pred_mask_binary, _ = predict_mask(model, image_normalized, img_size, args.threshold)
            
            # Save binary mask
            mask_output_path = os.path.join(masks_dir, f"{filename}_mask.png")
            cv2.imwrite(mask_output_path, pred_mask_binary.squeeze() * 255)
            
            # Evaluation with ground truth if available
            dice_score = None
            mask_path = None
            
            if eval_mode:
                # Try to find matching mask by filename
                base_name = os.path.basename(image_path)
                # Try several possible mask naming conventions
                potential_mask_names = [
                    base_name,  # Same name
                    base_name.replace('.', 'm.'),  # name.jpg -> namem.jpg
                    filename + 'm.png',  # name.jpg -> namem.png
                    filename + '_mask.png'  # name.jpg -> name_mask.png
                ]
                
                for mask_name in potential_mask_names:
                    if mask_name in mask_paths:
                        mask_path = mask_paths[mask_name]
                        break
                    
                if mask_path:
                    _, true_mask_norm = preprocess_mask(mask_path, img_size)
                    true_mask_tensor = np.expand_dims(true_mask_norm, axis=(0, -1))
                    pred_mask_tensor = np.expand_dims(pred_mask_binary.squeeze(), axis=(0, -1))
                    dice_score = tf.keras.backend.eval(dice_coefficient(true_mask_tensor, pred_mask_tensor))
            
            # Create and save visualization
            if eval_mode and mask_path:
                true_mask, _ = preprocess_mask(mask_path, img_size)
                save_visualization(
                    original_image, 
                    pred_mask_binary, 
                    true_mask, 
                    viz_dir, 
                    filename
                )
            else:
                save_visualization(
                    original_image, 
                    pred_mask_binary, 
                    output_path=viz_dir, 
                    filename=filename
                )
            
            # Create and save overlay if requested
            if args.save_overlay:
                save_overlay(original_image, pred_mask_binary, overlay_dir, filename)
            
            # Collect results
            result = {
                "filename": os.path.basename(image_path),
                "mask_path": mask_output_path,
                "dice_score": dice_score if dice_score is not None else "N/A",
                "has_lesion": np.any(pred_mask_binary > 0),
                "lesion_percentage": np.mean(pred_mask_binary) * 100
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output, "segmentation_results.csv"), index=False)
    
    # Print summary statistics
    if eval_mode:
        valid_dice_scores = [r["dice_score"] for r in results if r["dice_score"] != "N/A"]
        if valid_dice_scores:
            avg_dice = sum(valid_dice_scores) / len(valid_dice_scores)
            print(f"Average Dice Coefficient: {avg_dice:.4f}")
            print(f"Min Dice: {min(valid_dice_scores):.4f}, Max Dice: {max(valid_dice_scores):.4f}")
        else:
            print("No valid dice scores calculated")
    
    num_with_lesions = sum(1 for r in results if r["has_lesion"])
    print(f"Images with detected lesions: {num_with_lesions}/{len(results)} ({num_with_lesions/len(results)*100:.1f}%)")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
