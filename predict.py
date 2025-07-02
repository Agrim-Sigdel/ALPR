import tensorflow as tf
import numpy as np
import os
import glob
import json

# --- Parameters ---
MODEL_PATH = 'my_character_recognition_model.h5'
INDICES_PATH = 'class_indices.json'
TEST_FOLDER_PATH = './segmented_characters'
IMAGE_SIZE = (32, 32)

# --- 1. Load the trained model ---
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# --- 2. Load the class indices ---
print(f"Loading class indices from: {INDICES_PATH}")
with open(INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# Flip the dictionary to map from index to class name
idx_to_class = {v: k for k, v in class_indices.items()}

# --- 3. Find all images in the test folder ---
image_paths = glob.glob(os.path.join(TEST_FOLDER_PATH, '*.[pP][nN][gG]'))
image_paths.extend(glob.glob(os.path.join(TEST_FOLDER_PATH, '*.[jJ][pP][gG]')))
image_paths.extend(glob.glob(os.path.join(TEST_FOLDER_PATH, '*.[jJ][pP][eE][gG]')))

# --- ADD THIS LINE TO SORT THE FILES ---
image_paths.sort()

if not image_paths:
    print(f"\n❌ No images found in the folder: '{TEST_FOLDER_PATH}'")
else:
    print(f"\nFound {len(image_paths)} images to test. Starting predictions...\n")

    # --- 4. Loop through images and predict ---
    for image_path in image_paths:
        image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array, verbose=0)
        
        predicted_index = np.argmax(predictions[0])
        predicted_class = idx_to_class[predicted_index]
        confidence = 100 * np.max(predictions[0])
        
        filename = os.path.basename(image_path)
        print(f"File: {filename:<20} ->  Predicted: '{predicted_class}' ({confidence:.2f}% confidence)")

    print("\n--- All predictions finished. ---")