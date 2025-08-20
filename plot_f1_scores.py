import matplotlib.pyplot as plt
import numpy as np

# F1-Scores from my classification report
f1_scores = {
    'daisy': 0.92,
    'dandelion': 0.94,
    'rose': 0.89,
    'sunflower': 0.92,
    'tulip': 0.90
}

class_labels = list(f1_scores.keys())
scores = list(f1_scores.values())

plt.figure(figsize=(10, 6))
plt.bar(class_labels, scores, color='skyblue')
plt.xlabel('Flower Class')
plt.ylabel('F1-Score')
plt.title('Per-Class F1-Scores')
plt.ylim(0.8, 1.0) # Set y-axis limit for better visualization of differences
for i, score in enumerate(scores):
    plt.text(i, score + 0.005, f'{score:.2f}', ha='center', va='bottom')
plt.show()
