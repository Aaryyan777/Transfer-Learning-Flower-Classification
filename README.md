# Flower Classification with CNNs

## Problem Statement

Accurate and automated identification of plant species is crucial for various applications, including biodiversity monitoring, agricultural management, and ecological research. This project addresses the challenge of classifying flower images into distinct species using deep learning techniques, aiming to develop a robust and highly accurate automated system. The goal is to demonstrate the effectiveness of Convolutional Neural Networks (CNNs) in visual recognition tasks, providing a foundation for real-world applications like mobile flower identification apps.

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images of flowers. It utilizes the "Flowers Recognition (by Alexander)" dataset from Kaggle.

## Project Structure

```
flower-cnn/
├── data/                 # Contains the flower image dataset (daisy, dandelion, rose, sunflower, tulip)
├── src/
│   ├── train.py          # Script for training the CNN model
│   └── predict.py        # Script for making predictions with the trained model
├── flower_classification_model.h5 # Trained model file
└── requirements.txt      # Python dependencies
```

## Setup and Installation

1.  **Clone the repository (or create the project structure manually):**

    ```bash
    git clone <repository_url>
    cd flower-cnn
    ```

2.  **Download the Dataset:**

    Download the "Flowers Recognition (by Alexander)" dataset from Kaggle. This dataset comprises approximately 4,300 images categorized into 5 distinct flower classes: daisy, dandelion, rose, sunflower, and tulip. Extract the contents (the `daisy`, `dandelion`, `rose`, `sunflower`, and `tulip` directories) directly into the `flower-cnn/data` directory.

    The `data` directory structure should look like this:

    ```
    flower-cnn/data/
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the CNN model, run the `train.py` script:

```bash
python src/train.py
```

This script will:

*   Load and preprocess the image data.
*   Apply data augmentation techniques.
*   Build a CNN architecture.
*   Train the model for 50 epochs.
*   Save the trained model as `flower_classification_model.h5` in the project root directory.

### Making Predictions

To use the trained model for predictions, run the `predict.py` script:

```bash
python src/predict.py
```

**Note:** You will need to uncomment and modify the `test_image_path` variable in `predict.py` to point to an actual image file you want to classify.

## Methodology

This project employs a sophisticated deep learning methodology to achieve high accuracy in flower classification, leveraging transfer learning and progressive fine-tuning:

1.  **Transfer Learning with MobileNetV2:** We utilize a pre-trained MobileNetV2 model, initialized with weights from the ImageNet dataset, as a powerful feature extractor. This approach significantly reduces training time and improves performance, especially with limited datasets.

2.  **Focal Loss Implementation:** Instead of traditional categorical cross-entropy, the model is trained with a custom **Focal Loss** function. This addresses potential class imbalance by down-weighting easy examples and focusing training on hard-to-classify samples, leading to more robust learning.

3.  **Progressive Resizing and Enhanced Data Augmentation:** The training is conducted in two progressive stages:
    *   **Stage 1 (128x128 images):** The model is initially trained on smaller `128x128` images. This allows for rapid learning of general features and acts as a strong regularizer.
    *   **Stage 2 (150x150 images):** The model is then fine-tuned on higher-resolution `150x150` images, with weights transferred from Stage 1. Crucially, this stage incorporates **extensive data augmentation** (including rotations, width/height shifts, shear, zoom, brightness adjustments, channel shifts, and horizontal flips) to enhance generalization and prevent overfitting.

4.  **Aggressive Fine-tuning:** In Stage 2, all layers of the MobileNetV2 base model are unfrozen, allowing for comprehensive fine-tuning of the entire network to the specific characteristics of the flower dataset.

5.  **Adaptive Learning Rate and Early Stopping:** The training process is optimized with advanced callbacks:
    *   `ReduceLROnPlateau`: Dynamically reduces the learning rate by a factor of 0.1 when the validation loss plateaus, helping the model converge more effectively and escape local minima.
    *   `EarlyStopping`: Monitors the validation loss and halts training if no improvement is observed for a set number of epochs, preventing overfitting and saving computational resources.

## Results

Our iterative approach and refined training strategy led to significant improvements in model performance. Initially, a simpler CNN model yielded accuracies around 70%. Through the application of transfer learning with MobileNetV2, progressive resizing, Focal Loss, and enhanced data augmentation, we successfully boosted the model's validation accuracy to over 90%.

After implementing the advanced training strategy with enhanced data augmentation in Stage 2 and unfreezing all layers in Stage 2, the model achieved the following performance on the validation set:

*   **Validation Accuracy:** **91.63%**
*   **Validation Loss (Focal Loss):** **0.0301**

**Per-Class F1-Scores:**

| Flower      | F1-Score |
|-------------|----------|
| Daisy       | 0.92     |
| Dandelion   | 0.94     |
| Rose        | 0.89     |
| Sunflower   | 0.92     |
| Tulip       | 0.90     |

**Per-Class Precision and Recall:**

| Class       | Precision | Recall |
| :---------- | :-------- | :----- |
| Daisy       | 0.92      | 0.92   |
| Dandelion   | 0.96      | 0.92   |
| Rose        | 0.90      | 0.88   |
| Sunflower   | 0.91      | 0.93   |
| Tulip       | 0.89      | 0.92   |

This represents a significant achievement, surpassing the 90% accuracy target. The model stopped early at Epoch 56, with the best weights restored from Epoch 46, demonstrating effective convergence and generalization.

### Challenges and Improvements

Throughout the development, we addressed several key challenges:

*   **Accuracy Improvement:** Progressed from an initial accuracy of approximately 70% (with a simpler CNN) to a robust 91.63% validation accuracy by implementing transfer learning, progressive resizing, and optimized training strategies.
*   **Rose F1-Score Enhancement:** The F1-score for the 'Rose' class, which was initially around 0.83, improved to 0.89, indicating better classification for this specific category.
*   **Overfitting Mitigation:** The generalization gap, a key indicator of overfitting, was significantly reduced from an initial estimate of ~7.23% to a much healthier **2.27%**, demonstrating the effectiveness of our regularization techniques and data augmentation strategies.

### Generalization Gap

The generalization gap is the difference between the training accuracy and the validation accuracy, indicating how well the model generalizes to unseen data. A smaller gap suggests better generalization.

*   **Training Accuracy:** 93.90%
*   **Validation Accuracy:** 91.63%

**Generalization Gap:** 93.90% - 91.63% = **2.27%**

This small generalization gap indicates that the model is generalizing well and is not significantly overfitting to the training data.

## Visualizations

To provide deeper insights into the model's performance and training dynamics, the following visualizations have been generated:

*   **Confusion Matrix:** Illustrates the performance of the classification model, showing the number of correct and incorrect predictions for each class. (Generated by `plot_confusion_matrix.py`)
*   **Per-Class F1-Scores Bar Chart:** Provides a clear comparison of the F1-scores for each flower class, highlighting strengths and areas for potential improvement. (Generated by `plot_f1_scores.py`)
*   **Learning Curves (Accuracy and Loss):** Plots the training and validation accuracy and loss over epochs, crucial for diagnosing overfitting, underfitting, and convergence. (Generated by `plot_learning_curves.py`)
*   **Class Distribution in Dataset:** Visualizes the number of images per class, helping to identify potential class imbalances. (Generated by `plot_class_distribution.py`)

These visualizations offer a comprehensive overview of the model's behavior and performance.

## Future Work

To further enhance this project and explore its practical applications, we propose the following future work:

*   **Hyperparameter Tuning:** Systematically optimize hyperparameters (e.g., learning rates, batch sizes, optimizer configurations, augmentation parameters) using techniques like grid search, random search, or Bayesian optimization to squeeze out even higher performance.
*   **Deployment as a Flower Identification App:** Develop a user-friendly application (e.g., a web application using Flask/FastAPI or a mobile app) that allows users to upload flower images and receive instant classification results. This would demonstrate the practical utility of the trained model.
*   **Advanced Model Architectures:** Experiment with other state-of-the-art CNN architectures (e.g., EfficientNet, ResNeXt, Vision Transformers) or explore hybrid models to potentially achieve even higher accuracy and efficiency.
*   **Model Interpretability (Grad-CAM/SHAP):** Implement techniques like Grad-CAM (for CNNs) or SHAP (for more general ML models) to visualize and explain model predictions, enhancing trust and understanding of the model's decision-making process. This is particularly useful for analyzing misclassifications or confirming correct predictions.
*   **Ensemble Methods:** Combine predictions from multiple diverse models to further boost overall accuracy and robustness.
*   **Test-Time Augmentation (TTA):** Implement TTA during inference by making predictions on multiple augmented versions of a test image and averaging the results for a more robust final prediction.
