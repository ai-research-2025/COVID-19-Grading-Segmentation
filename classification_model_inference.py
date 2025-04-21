import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import pickle
import os
import argparse
from PIL import Image
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='COVID Classification Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file (.pth)')
    parser.add_argument('--feature-selector', type=str, required=True, help='Path to feature selector file (.pkl)')
    parser.add_argument('--input', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--output', type=str, default=None, help='Path to save results (CSV format)')
    return parser.parse_args()

# Define Classification Model (same as training script)
class ClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2)  # Binary Classification
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def predict_single_image(image_path, model, selector, resnet, transform, device):
    """Predict class for a single image"""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = resnet(img_tensor).view(-1).cpu().numpy()
            features_selected = selector.transform(features.reshape(1, -1))
            input_tensor = torch.tensor(features_selected, dtype=torch.float32).to(device)
            output = model(input_tensor)
            # Get both prediction and probabilities
            prediction = torch.argmax(output, dim=1).item()
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            
        result = {
            'image_path': image_path,
            'prediction': prediction,
            'class_name': 'COVID' if prediction == 1 else 'Normal',
            'confidence': float(probabilities[prediction])
        }
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return {
            'image_path': image_path,
            'prediction': None,
            'class_name': 'Error',
            'confidence': 0.0
        }

def main():
    args = parse_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load feature selector
    print(f"Loading feature selector from {args.feature_selector}")
    with open(args.feature_selector, "rb") as f:
        selector = pickle.load(f)
    
    # Load the model
    print(f"Loading model from {args.model}")
    input_dim = selector.get_support(indices=True).shape[0]  # Get number of selected features
    model = ClassificationModel(input_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Load ResNet-18 (same as training)
    print("Loading pretrained ResNet-18...")
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
    resnet.to(device)
    resnet.eval()
    
    # Set up transforms (same as training but without augmentations)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get list of images to process
    if os.path.isdir(args.input):
        # Process all images in directory
        image_paths = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for root, _, files in os.walk(args.input):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(root, file))
        print(f"Found {len(image_paths)} images to process in {args.input}")
    else:
        # Process single image
        image_paths = [args.input]
        print(f"Processing single image: {args.input}")
    
    # Process images
    results = []
    for image_path in tqdm(image_paths):
        result = predict_single_image(
            image_path, model, selector, resnet, transform, device
        )
        results.append(result)
        if len(image_paths) == 1:  # If just one image, print result immediately
            print(f"Prediction: {result['class_name']} (Confidence: {result['confidence']:.4f})")
    
    # Save results to CSV if output path is provided
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    elif len(image_paths) > 1:
        # Show summary of results
        covid_count = sum(1 for r in results if r['class_name'] == 'COVID')
        normal_count = sum(1 for r in results if r['class_name'] == 'Normal')
        error_count = sum(1 for r in results if r['class_name'] == 'Error')
        
        print("\nSummary:")
        print(f"- COVID: {covid_count}")
        print(f"- Normal: {normal_count}")
        print(f"- Errors: {error_count}")

if __name__ == "__main__":
    main()
