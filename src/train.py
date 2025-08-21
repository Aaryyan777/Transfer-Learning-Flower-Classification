
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from focal_loss import focal_loss # Import the custom loss function
import os

# Define paths
data_dir = r"C:\Users\DELL\flower-cnn\data"
batch_size = 32

# --- Stage 1: Progressive Resizing (128x128) ---
print("--- Starting Stage 1: Training on 128x128 images ---")
img_height_s1, img_width_s1 = 128, 128

# Data Augmentation for Stage 1
train_datagen_s1 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.8, 1.2],
    channel_shift_range=50.0,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
validation_datagen_s1 = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator_s1 = train_datagen_s1.flow_from_directory(
    data_dir,
    target_size=(img_height_s1, img_width_s1),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator_s1 = validation_datagen_s1.flow_from_directory(
    data_dir,
    target_size=(img_height_s1, img_width_s1),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build Model for Stage 1
base_model_s1 = MobileNetV2(input_shape=(img_height_s1, img_width_s1, 3), include_top=False, weights='imagenet')
base_model_s1.trainable = True
for layer in base_model_s1.layers[:-40]: # Unfreeze last 40 layers
    layer.trainable = False

model_s1 = Sequential([
    base_model_s1,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator_s1.num_classes, activation='softmax')
])

# Compile with Focal Loss
model_s1.compile(optimizer=Adam(learning_rate=1e-4), # Start with a slightly higher LR
                 loss=focal_loss(gamma=2.0, alpha=0.25),
                 metrics=['accuracy'])

# Callbacks for Stage 1
reduce_lr_s1 = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
early_stopping_s1 = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# Train Stage 1
model_s1.fit(
    train_generator_s1,
    epochs=30, # Train for fewer epochs on smaller images
    validation_data=validation_generator_s1,
    callbacks=[reduce_lr_s1, early_stopping_s1]
)

# --- Stage 2: Progressive Resizing (150x150) ---
print("\n--- Starting Stage 2: Fine-tuning on 150x150 images ---")
img_height_s2, img_width_s2 = 150, 150

# Data Augmentation for Stage 2
train_datagen_s2 = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_datagen_s2 = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator_s2 = train_datagen_s2.flow_from_directory(
    data_dir,
    target_size=(img_height_s2, img_width_s2),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator_s2 = validation_datagen_s2.flow_from_directory(
    data_dir,
    target_size=(img_height_s2, img_width_s2),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build and load weights into the new model
base_model_s2 = MobileNetV2(input_shape=(img_height_s2, img_width_s2, 3), include_top=False, weights='imagenet')
base_model_s2.trainable = True
base_model_s2.trainable = True # Unfreeze all layers

model = Sequential([
    base_model_s2,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator_s2.num_classes, activation='softmax')
])

# Load weights from the best checkpoint of Stage 1 model
# Note: This is a simplified way. For different architectures, you might need to transfer layer by layer.
model.set_weights(model_s1.get_weights())

# Re-compile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-6),
              loss=focal_loss(gamma=2.0, alpha=0.25),
              metrics=['accuracy'])

# Callbacks for Stage 2
reduce_lr_s2 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=1e-7, verbose=1)
early_stopping_s2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train Stage 2
model.fit(
    train_generator_s2,
    epochs=100, # Continue training for more epochs
    validation_data=validation_generator_s2,
    callbacks=[reduce_lr_s2, early_stopping_s2]
)

# Save the final model
model.save(r"C:\Users\DELL\flower-cnn\flower_classification_model.h5")

print("Model training complete and saved as flower_classification_model.h5")

# Evaluate the model and print additional metrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("Evaluating model on validation data...")

# Collect true labels and predictions
validation_generator_s2.reset()
y_true = []
y_pred = []

num_batches = (validation_generator_s2.samples + batch_size - 1) // batch_size

for i in range(num_batches):
    x_val, y_val = next(validation_generator_s2)
    y_true.extend(np.argmax(y_val, axis=1))
    y_pred.extend(np.argmax(model.predict(x_val), axis=1))

class_labels = list(validation_generator_s2.class_indices.keys())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

val_loss, val_accuracy = model.evaluate(validation_generator_s2, verbose=0)
print(f"\nValidation Cross-Entropy Loss: {val_loss:.4f}")
