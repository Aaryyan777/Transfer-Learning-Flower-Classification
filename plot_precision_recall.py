import matplotlib.pyplot as plt
import numpy as np

# Data from your classification report
class_metrics = {
    'daisy': {'precision': 0.92, 'recall': 0.92, 'f1-score': 0.92},
    'dandelion': {'precision': 0.96, 'recall': 0.92, 'f1-score': 0.94},
    'rose': {'precision': 0.90, 'recall': 0.88, 'f1-score': 0.89},
    'sunflower': {'precision': 0.91, 'recall': 0.93, 'f1-score': 0.92},
    'tulip': {'precision': 0.89, 'recall': 0.92, 'f1-score': 0.90}
}

class_labels = list(class_metrics.keys())
precision_scores = [class_metrics[c]['precision'] for c in class_labels]
recall_scores = [class_metrics[c]['recall'] for c in class_labels]

x = np.arange(len(class_labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, precision_scores, width, label='Precision', color='skyblue')
rects2 = ax.bar(x + width/2, recall_scores, width, label='Recall', color='lightcoral')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score')
ax.set_title('Per-Class Precision and Recall')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()
ax.set_ylim(0.8, 1.0) # Adjust y-axis for better visualization

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()
