
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3
NUM_CLASSES = 43
EPOCHS = 15
BATCH_SIZE = 32

def load_data(data_dir):
    images = []
    labels = []
    
    print("Loading Data... This might take a minute.")
    
    # loop through all 43 classes folders
    for i in range(NUM_CLASSES):
        path = os.path.join(data_dir, 'Train', str(i))
        img_folder = os.listdir(path)
        
        for img_name in img_folder:
            try:
                # Read Image
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                
                # Resize to 32x32
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                
                # Add to list
                images.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize pixel values (0-255 -> 0-1)
    images = images / 255.0
    
    return images, labels

def main():
    # Load and Preprocess
    X, y = load_data(DATA_DIR)
    
    # Split Data (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encode labels (e.g., Class 3 becomes [0,0,0,1,0...])
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")

    # Data Augmentation (Fixes Class Imbalance & Overfitting)
    # This creates new variations of images (zoomed, rotated) on the fly
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=False, # Traffic signs usually aren't flipped horizontally!
        fill_mode="nearest"
    )
    
    # Build and Train Model
    model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), num_classes=NUM_CLASSES)
    
    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val))
    
    # Save the Model
    if not os.path.exists('../models'):
        os.makedirs('../models')
    model.save('../models/traffic_classifier.h5')
    print("Model saved to models/traffic_classifier.h5")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

if __name__ == '__main__':
    main()