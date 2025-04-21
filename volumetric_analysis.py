import os
import cv2
import numpy as np
import argparse
import logging

def preprocess_ct_images(ct_path, img_size=(256, 256)):
    """Preprocess CT images and sort by view type"""
    axial_images = []
    coronal_images = []
    sagittal_images = []
    
    for img_name in sorted(os.listdir(ct_path)):
        img_path = os.path.join(ct_path, img_name)
        
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue
            
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize and normalize (as per original code)
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0

            # Categorize views based on filename
            if "axial" in img_name.lower():
                axial_images.append(img)
            elif "coronal" in img_name.lower():
                coronal_images.append(img)
            elif "sagittal" in img_name.lower():
                sagittal_images.append(img)
                
        except Exception as e:
            logging.error(f"Error processing {img_name}: {str(e)}")
            continue

    return np.array(axial_images), np.array(coronal_images), np.array(sagittal_images)

def compute_volumetric_scores(images):
    """Calculate scores as sum of intensities divided by total pixels (mean intensity)"""
    scores = []
    for img in images:
        score = np.sum(img) / (img.shape[0] * img.shape[1])
        scores.append(score)
    return np.array(scores)

def assign_severity_labels(scores):
    """Classify severity based on original thresholds"""
    labels = []
    for score in scores:
        if score < 0.3:  # Healthy
            labels.append("Healthy")
        elif score < 0.6:  # Mild
            labels.append("Mild")
        else:  # Moderate
            labels.append("Moderate")
    return np.array(labels)

def main():
    parser = argparse.ArgumentParser(description='COVID-19 Volumetric Analysis')
    parser.add_argument('--ct_path', type=str, required=True,
                       help='Path to directory containing CT images')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for analysis results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'analysis.log'), 
                        level=logging.INFO)

    # Process images
    axial, coronal, sagittal = preprocess_ct_images(args.ct_path)
    
    # Compute scores for each view
    views = {
        'Axial': axial,
        'Coronal': coronal,
        'Sagittal': sagittal
    }

    results = {}
    for view_name, images in views.items():
        if len(images) == 0:
            logging.warning(f"No {view_name} images found")
            continue
            
        scores = compute_volumetric_scores(images)
        labels = assign_severity_labels(scores)
        
        # Save results
        results[view_name] = {
            'scores': scores,
            'labels': labels
        }
        
        # Print summary
        print(f"\n{view_name} View Analysis:")
        print(f"Total Images: {len(images)}")
        print(f"Healthy: {np.sum(labels == 'Healthy')}")
        print(f"Mild: {np.sum(labels == 'Mild')}")
        print(f"Moderate: {np.sum(labels == 'Moderate')}")

    # Save full results to numpy file
    np.save(os.path.join(args.output_dir, 'volumetric_results.npy'), results)
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
