import tensorflow as tf
from tensorflow.keras import layers, models

# --- Parameters ---
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
DATA_DIR = 'ocr_dataset' # The path to your dataset folder
EPOCHS = 15 # Start with 15-20 and see how the accuracy improves

# --- Step 1: Create Training & Validation Datasets ---
# This re-loads the data to ensure the script is self-contained.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2, # Use 20% of the data for validation
    subset="training",
    seed=123, # Seed for reproducibility
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical' # Important for multi-class classification
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Get the number of classes that Keras found (should be 71)
num_classes = len(train_dataset.class_names)
print(f"Number of classes found: {num_classes}")


# --- Step 2: Build Your CNN Model ---
model = models.Sequential([
    # Normalize pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten the results to feed into a dense layer
    layers.Flatten(),

    # A standard dense layer for classification
    layers.Dense(128, activation='relu'),

    # The final output layer must have a neuron for each class
    layers.Dense(num_classes, activation='softmax')
])

# Print a summary of the model
model.summary()

# --- Step 3: Compile and Train the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Use this for multi-class classification
    metrics=['accuracy']
)

print("\n--- Starting Training ---")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("--- Training Finished ---")


# --- Step 4: Save Your Trained Model ---
model.save('my_character_recognition_model.h5')
print("✅ Model saved successfully as my_character_recognition_model.h5")