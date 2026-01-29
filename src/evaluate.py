import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
# Automatically find paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'traffic_classifier.h5')

def evaluate_model():
    print("Loading Test Data...")
    
    # 1. Load the CSV to get the ground truth
    csv_path = os.path.join(DATA_DIR, 'Test.csv')
    test_df = pd.read_csv(csv_path)
    
    # 2. Load and Preprocess Test Images
    test_images = []
    test_labels = test_df['ClassId'].values
    image_paths = test_df['Path'].values

    for img_path in image_paths:
        full_path = os.path.join(DATA_DIR, img_path)
        image = cv2.imread(full_path)
        image = cv2.resize(image, (32, 32)) # Must match training size
        test_images.append(image)

    # Convert to numpy array and normalize
    X_test = np.array(test_images)
    X_test = X_test / 255.0

    print(f"Loaded {len(X_test)} test images.")

    # 3. Load Model and Predict
    print("Loading Model...")
    model = load_model(MODEL_PATH)
    
    print("Predicting...")
    pred_probs = model.predict(X_test)
    pred_classes = np.argmax(pred_probs, axis=1)

    # 4. Calculate Accuracy
    acc = accuracy_score(test_labels, pred_classes)
    print(f"\nFINAL TEST ACCURACY: {acc * 100:.2f}%")
    
    # Save a report to see which signs it fails on
    print(classification_report(test_labels, pred_classes))

if __name__ == '__main__':
    evaluate_model()