import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
    .prediction-card {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 1rem;
        border-left: 5px solid #007bff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
        margin: 0.5rem 0;
    }
    .confidence-score {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .high-conf { color: #28a745; }
    .low-conf { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# --- CLASS DICTIONARY ---
CLASSES = { 
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 
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
    42:'End no passing veh > 3.5 tons' 
}

@st.cache_resource
def load_my_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'traffic_classifier.h5')
    return load_model(MODEL_PATH)

with st.sidebar:
    st.title("Configuration")
    st.markdown("---")
    st.write("**Model Architecture**")
    st.code("CNN (3 Conv Layers)", language="text")
    st.write("**Input Resolution**") 
    st.code("32 x 32 px", language="text")
    st.markdown("---")
    st.caption("Traffic Sign Recognition Project v1.0")

# MAIN INTERFACE
st.title("Traffic Sign Classification")
st.markdown("Upload a traffic sign image to generate a prediction.")

try:
    model = load_my_model()
    
    # File Uploader
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Create two columns for a dashboard feel
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### Source Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Analysis Results")
            
            if st.button("Process Image"):
                with st.spinner('Analyzing pixel data...'):
                    # Resize the PIL image (RGB)
                    img = image.resize((32, 32))
                    
                    img_array = np.array(img)
                    
                    # Convert RGB to BGR
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array)
                    class_id = np.argmax(predictions)
                    confidence = np.max(predictions)
                    label = CLASSES.get(class_id, "Unknown Label")

                    # Determine color class based on confidence
                    conf_color_class = "high-conf" if confidence > 0.8 else "low-conf"
                    
                    # Display Result Card
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="metric-label">Identified Sign</div>
                        <div class="metric-value">{label}</div>
                        <div class="metric-label" style="margin-top: 1rem;">Model Confidence</div>
                        <div class="confidence-score {conf_color_class}">{confidence * 100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if confidence < 0.6:
                        st.info("Note: Confidence is low. Ensure the image is clear and well-lit.")

except Exception as e:
    st.error(f"System Error: Failed to load model. {e}")