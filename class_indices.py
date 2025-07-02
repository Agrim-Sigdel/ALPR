import tensorflow as tf
import json

print("Generating class indices file...")

# --- Parameters ---
DATA_DIR = 'ocr_dataset'

# Load the dataset to get its configuration
dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(32, 32),
    batch_size=1
)

# --- Correct way to create the class_indices dictionary ---
# 1. Get the list of class names from the dataset.
class_names = dataset.class_names

# 2. Build the dictionary by mapping each class name to its index.
class_indices = {name: i for i, name in enumerate(class_names)}


# --- Save the Class Indices dictionary to a JSON file ---
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print(f"✅ Class indices for {len(class_indices)} classes saved to 'class_indices.json'")