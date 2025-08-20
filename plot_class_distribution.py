import matplotlib.pyplot as plt
import os

data_dir = r"C:\Users\DELL\flower-cnn\data"
class_names = sorted(os.listdir(data_dir))

class_counts = {}
for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

labels = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='lightgreen')
plt.xlabel('Flower Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Dataset')
plt.xticks(rotation=45, ha='right')
for i, count in enumerate(counts):
    plt.text(i, count + 10, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.show()
