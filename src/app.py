import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'traffic_classifier.h5')

# Dictionary to name the classes
CLASSES = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 9:'No passing', 
            10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 
            12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 
            19:'Dangerous curve left', 20:'Dangerous curve right', 
            21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 
            24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 
            27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing', 
            30:'Beware of ice/snow', 31:'Wild animals crossing', 
            32:'End speed + passing limits', 33:'Turn right ahead', 
            34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 
            37:'Go straight or left', 38:'Keep right', 39:'Keep left', 
            40:'Roundabout mandatory', 41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

def load_trained_model():
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found! Please train the model first.")
        return None
    return load_model(MODEL_PATH)

def predict_and_display(image_path, model):
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not open image. Try a standard JPG or PNG.")
        return

    # Resize to 32x32 (what the model expects)
    model_input = cv2.resize(img, (32, 32))
    model_input = model_input / 255.0
    model_input = np.expand_dims(model_input, axis=0)
    
    prediction = model.predict(model_input)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    label = CLASSES.get(class_id, "Unknown")
    
    print(f"File: {os.path.basename(image_path)}")
    print(f"Prediction: {label} ({confidence * 100:.2f}%)")

    # (Dashboard)
    canvas_h, canvas_w = 400, 600
    canvas = np.ones((canvas_h, canvas_w, 3), dtype="uint8") * 240 # Gray

    # Resize original image for display (Max 250px tall)
    h, w = img.shape[:2]
    scale = min(250/h, 250/w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_display = cv2.resize(img, (new_w, new_h))
    
    # Center the image
    y_offset = 30
    x_offset = (canvas_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_display
    
    # Draw Border
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset+new_w, y_offset+new_h), (0,0,0), 2)

    # Text Setup
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Prediction Text
    text_size = cv2.getTextSize(label, font, 0.9, 1)[0]
    text_x = (canvas_w - text_size[0]) // 2
    cv2.putText(canvas, label, (text_x, y_offset + new_h + 50), font, 0.9, (20, 20, 20), 1)

    # Confidence Text
    conf_text = f"Confidence: {confidence * 100:.1f}%"
    text_size = cv2.getTextSize(conf_text, font, 0.7, 1)[0]
    text_x = (canvas_w - text_size[0]) // 2
    color = (0, 180, 0) if confidence > 0.8 else (0, 0, 200)
    cv2.putText(canvas, conf_text, (text_x, y_offset + new_h + 85), font, 0.7, color, 1)

    # Instruction Text
    cv2.putText(canvas, "Press any key to exit", (20, 380), font, 0.5, (100, 100, 100), 1)

    cv2.imshow("Traffic Sign Recognition App", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load model once
    model = load_trained_model()
    if model is None: return

    # Tkinter
    root = tk.Tk()
    root.withdraw() 

    print("Opening file dialog...")
    
    # Open the File Picker
    file_path = filedialog.askopenfilename(
        title="Select a Traffic Sign Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if file_path:
        predict_and_display(file_path, model)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()