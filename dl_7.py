
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Define input parameters
IMG_SIZE = (224, 224)
NUM_CLASSES = 2  # e.g., cats vs. dogs
BATCH_SIZE = 32

# --- 1. Simulate a Small Custom Dataset ---
# In a real scenario, use:
# tf.keras.preprocessing.image_dataset_from_directory()

print("Simulating dataset...")
# Create synthetic data (200 training, 50 validation samples)
X_train = np.random.rand(200, *IMG_SIZE, 3).astype('float32')
y_train = np.random.randint(0, NUM_CLASSES, 200)
X_val = np.random.rand(50, *IMG_SIZE, 3).astype('float32')
y_val = np.random.randint(0, NUM_CLASSES, 50)

# Apply typical ImageNet preprocessing (centering/scaling)
X_train = tf.keras.applications.resnet50.preprocess_input(X_train * 255.0)
X_val = tf.keras.applications.resnet50.preprocess_input(X_val * 255.0)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
print(f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}")

# --- 2. Load Pretrained Model (ResNet50) ---
print("\nLoading pretrained ResNet50 base...")

# Load ResNet50 weights pretrained on ImageNet
# include_top=False: Exclude the original 1000-class classification head
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# --- 3. Feature Extraction Phase (Freeze the entire base) ---
# Freeze the base model layers so their weights are NOT updated during the first training phase.
base_model.trainable = False
print("Base model frozen.")

# Create the full model: Base + New Classification Head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduces the 3D feature maps to a 1D vector
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') # Custom classification layer
])

# Compile the model for the feature extraction phase
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Phase 1: Feature Extraction (Training only the head) ---")
# Train only the top layers for a few epochs
history_fe = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE
)

# --- 4. Fine-Tuning Phase (Unfreeze top layers of the base) ---
print("\n--- Phase 2: Fine-Tuning (Unfreezing top ResNet blocks) ---")

# Unfreeze the base model
base_model.trainable = True
print(f"Base model layers: {len(base_model.layers)}")

# Freeze all layers UP TO a certain point (e.g., the last few convolutional blocks)
# ResNet50 has many layers; we typically freeze the early ones.
# Let's freeze the first 140 layers, leaving the last ~35 layers trainable.
fine_tune_at = 140

# Freeze all layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Layers from 0 to {fine_tune_at-1} are frozen. Layers from {fine_tune_at} onwards are trainable.")

# Compile the model with a very low learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), # Use a VERY small learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training the entire model with the new, small learning rate
history_ft = model.fit(
    X_train, y_train,
    epochs=10, # Total epochs is 5 (Phase 1) + 5 (Phase 2)
    initial_epoch=history_fe.epoch[-1] + 1, # Start from the next epoch after Phase 1
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE
)

# --- 5. Visualize Results (for a real dataset, this would show real trends) ---
print("\nTraining complete. Visualizing results.")

# Combine history for plotting
acc = history_fe.history['accuracy'] + history_ft.history['accuracy']
val_acc = history_fe.history['val_accuracy'] + history_ft.history['val_accuracy']
loss = history_fe.history['loss'] + history_ft.history['loss']
val_loss = history_fe.history['val_loss'] + history_ft.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.axvline(x=len(history_fe.epoch), color='r', linestyle='--', label='Fine-Tuning Start')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.axvline(x=len(history_fe.epoch), color='r', linestyle='--')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

print("\nModel Summary (Trainable vs. Non-Trainable parameters after fine-tuning):")
model.summary()

