import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import numpy as np
import pickle
import os
import argparse
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Train COVID classification model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save model and feature selector')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_features', type=int, default=446, help='Number of features to select')
    return parser.parse_args()

# Define Classification Model
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

# Extract Features Function
def extract_features(loader, resnet, device):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = resnet(imgs).view(imgs.size(0), -1).cpu().numpy()  # Flatten features
            features.extend(feats)
            labels.extend(lbls.numpy())
    return np.array(features, dtype=np.float32), np.array(labels)

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Pretrained ResNet-18
    print("Loading pretrained ResNet-18...")
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
    resnet.to(device)
    resnet.eval()

    # Data Transformations with Augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load Dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = datasets.ImageFolder(root=args.dataset, transform=transform)
    train_data, test_data = train_test_split(dataset, test_size=0.2, stratify=dataset.targets, random_state=42)
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Extract Features
    print("Extracting features...")
    X_train, y_train = extract_features(train_loader, resnet, device)
    X_test, y_test = extract_features(test_loader, resnet, device)

    # Feature Selection
    print(f"Selecting top {args.num_features} features...")
    selector = SelectKBest(f_classif, k=args.num_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Save Feature Selector
    selector_path = os.path.join(args.output, "feature_selector.pkl")
    with open(selector_path, "wb") as f:
        pickle.dump(selector, f)
    print(f"Feature selector saved to {selector_path}")

    # Initialize Model
    input_dim = X_train_selected.shape[1]
    model = ClassificationModel(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    # Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(X_train_selected) // args.batch_size

        for i in range(0, len(X_train_selected), args.batch_size):
            batch_end = min(i + args.batch_size, len(X_train_selected))
            inputs = torch.tensor(X_train_selected[i:batch_end], dtype=torch.float32).to(device)
            labels = torch.tensor(y_train[i:batch_end], dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save Model Weights
    model_path = os.path.join(args.output, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluation
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test_selected, dtype=torch.float32).to(device)
        predictions = model(test_inputs).argmax(dim=1).cpu().numpy()
    print(classification_report(y_test, predictions))
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
