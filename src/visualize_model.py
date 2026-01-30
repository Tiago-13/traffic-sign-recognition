import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'traffic_classifier.h5')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'assets')

def generate_diagram():
    # 1. Load the model
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Please train it first.")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)

    # 2. Ensure assets folder exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, 'model_architecture.png')

    # 3. Generate the plot
    print("Generating diagram...")
    try:
        plot_model(
            model,
            to_file=output_path,
            show_shapes=True,        # Shows (32, 32, 3) -> (30, 30, 32)
            show_layer_names=True,   # Shows "conv2d_1", "max_pooling"
            rankdir='TB',            # Top to Bottom
            expand_nested=True,
            dpi=96
        )
        print(f"✅ Success! Diagram saved to: {output_path}")
    except ImportError:
        print("❌ Error: Graphviz is not installed correctly.")
        print("Please download it from https://graphviz.org/download/ and add it to your PATH.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    generate_diagram()