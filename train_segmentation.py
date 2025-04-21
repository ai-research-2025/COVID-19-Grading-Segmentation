import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2
import argparse
import albumentations as A
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train COVID-19 CT Segmentation Model")
    parser.add_argument("--images", type=str, required=True, help="Directory containing image frames")
    parser.add_argument("--masks", type=str, required=True, help="Directory containing mask frames")
    parser.add_argument("--output", type=str, required=True, help="Directory to save trained model and results")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--img-size", type=int, default=128, help="Image size for training (default: 128)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    return parser.parse_args()

# Dice coefficient function
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Combined Dice and Binary Cross Entropy loss
def dice_bce_loss(y_true, y_pred):
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return 0.5 * dice_loss + 0.5 * bce_loss  # Balance both losses

# Data Generator class
class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, img_size, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        
        # Data augmentation using Albumentations
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.ElasticTransform(p=0.2),
            A.GaussianBlur(p=0.1)
        ])

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images, batch_masks = [], []

        for i in indices:
            img = cv2.imread(self.image_paths[i])
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Unable to read image {self.image_paths[i]}")
                continue
                
            if mask is None:
                print(f"Warning: Unable to read mask {self.mask_paths[i]}")
                continue
                
            img = cv2.resize(img, self.img_size) / 255.0
            mask = cv2.resize(mask, self.img_size) / 255.0

            if self.augment:
                augmented = self.augmentation(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]

            batch_images.append(img)
            batch_masks.append(np.expand_dims(mask, axis=-1))

        return np.array(batch_images), np.array(batch_masks)

# Improved U-Net Model
def create_model(input_size):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def encoder_block(x, filters):
        x = conv_block(x, filters)
        p = layers.MaxPooling2D((2, 2))(x)
        p = layers.Dropout(0.1)(p)
        return x, p

    def decoder_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    # Encoder
    s1, p1 = encoder_block(inputs, 16)
    s2, p2 = encoder_block(p1, 32)
    s3, p3 = encoder_block(p2, 64)
    s4, p4 = encoder_block(p3, 128)

    # Bottleneck
    b = conv_block(p4, 256)

    # Decoder
    d1 = decoder_block(b, s4, 128)
    d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32)
    d4 = decoder_block(d3, s1, 16)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=dice_bce_loss, metrics=[dice_coefficient])
    return model

def main():
    # Parse arguments
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

    print(f"Loading images from {args.images}")
    print(f"Loading masks from {args.masks}")
    
    # Load file paths
    image_paths = []
    mask_paths = []
    
    for img_file in sorted(os.listdir(args.images)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            image_paths.append(os.path.join(args.images, img_file))
    
    for mask_file in sorted(os.listdir(args.masks)):
        if mask_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            mask_paths.append(os.path.join(args.masks, mask_file))
    
    if len(image_paths) != len(mask_paths):
        print(f"Warning: Number of images ({len(image_paths)}) does not match number of masks ({len(mask_paths)})")
    
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    # Check that we have matching images and masks
    if len(image_paths) == 0 or len(mask_paths) == 0:
        print("Error: No images or masks found. Please check your directories.")
        return
    
    # Split into training & validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=args.val_split, random_state=42
    )
    
    print(f"Training on {len(train_images)} images, validating on {len(val_images)} images")
    
    img_size = (args.img_size, args.img_size)
    
    # Create data generators
    train_gen = DataGenerator(train_images, train_masks, batch_size=args.batch_size, img_size=img_size, augment=True)
    val_gen = DataGenerator(val_images, val_masks, batch_size=args.batch_size, img_size=img_size, augment=False)
    
    # Create model
    model = create_model(args.img_size)
    model.summary()
    
    # Callbacks for better training
    model_path = os.path.join(args.output, "best_model.keras")
    history_path = os.path.join(args.output, "training_history.npy")
    plot_path = os.path.join(args.output, "training_plot.png")
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
    ]
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=args.epochs, 
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(args.output, "final_model.keras")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training history
    np.save(history_path, history.history)
    print(f"Training history saved to {history_path}")
    
    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['dice_coefficient'], label='Train Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title('Dice Coefficient Over Epochs')
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")

if __name__ == "__main__":
    main()
