
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Load the trained model with custom objects for focal loss
from focal_loss import focal_loss
model = tf.keras.models.load_model(
    r'C:\Users\DELL\flower-cnn\flower_classification_model.h5',
    custom_objects={'focal_loss_fixed': focal_loss()}
)

# Define image dimensions from the model's input layer
img_height, img_width = model.input_shape[1], model.input_shape[2]

# Get class names from the training data directory
data_dir = r"C:\Users\DELL\flower-cnn\data"
class_names = sorted(os.listdir(data_dir))

# Define the Test-Time Augmentation (TTA) generator
tta_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    fill_mode='nearest'
)

import time

def predict_image_with_tta(img_path, num_augmentations=10):
    """
    Makes a prediction on a single image using Test-Time Augmentation (TTA).
    """
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Generate augmented images and stack them
    # Note: Rescaling is now done inside the augmentation loop
    augmented_images = []
    img_iterator = tta_datagen.flow(img_array, batch_size=1)
    
    # Add the original, un-augmented image first
    augmented_images.append(img_array[0] / 255.0)

    # Generate augmented versions
    for _ in range(num_augmentations - 1):
        augmented_batch = next(img_iterator)
        augmented_images.append(augmented_batch[0] / 255.0)

    augmented_images = np.array(augmented_images)

    # Get predictions for all augmented images
    predictions = model.predict(augmented_images)

    # Average the predictions
    avg_predictions = np.mean(predictions, axis=0)

    # Get the final class and confidence
    predicted_class_index = np.argmax(avg_predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(avg_predictions)

    return predicted_class_name, confidence

# Example usage (replace with a path to an actual image)
# You can pick an image from the dataset for testing
test_image_path = r'C:\Users\DELL\flower-cnn\data\rose\12240303_80d87f77a3_n.jpg'

start_time = time.time()
predicted_class, confidence = predict_image_with_tta(test_image_path)
end_time = time.time()
inference_time = end_time - start_time

print(f"The image is predicted to be a {predicted_class} with {confidence:.2f} confidence.")
print(f"Inference time: {inference_time:.4f} seconds.")


