import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# Reconstruct the confusion matrix from the provided output
conf_matrix_data = np.array([
    [140,   4,   2,   2,   4],
    [  8, 194,   0,   5,   3],
    [  1,   0, 138,   4,  13],
    [  2,   4,   2, 136,   2],
    [  2,   0,  11,   3, 180]
])

class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_data, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()