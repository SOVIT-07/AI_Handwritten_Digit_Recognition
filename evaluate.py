import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

MODEL_PATH = "models/mnist_cnn_final.keras"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nLoading MNIST test dataset...")

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255.0

x_test = np.expand_dims(x_test, axis=-1)

print(f"Test images: {x_test.shape}")

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nGenerating predictions...")

probabilities = model.predict(
    x_test,
    batch_size=128,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

print("Predictions completed.")

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n===================================")
print("       CLASSIFICATION METRICS")
print("===================================")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("===================================\n")

report = classification_report(
    y_test,
    y_pred,
    digits=4
)

print("Classification Report:")
print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as file:

    file.write("MNIST CNN Classification Report\n")
    file.write("================================\n\n")

    file.write(f"Weighted Precision: {precision:.4f}\n")
    file.write(f"Weighted Recall:    {recall:.4f}\n")
    file.write(f"Weighted F1-Score:  {f1:.4f}\n\n")

    file.write("Detailed Classification Report:\n")
    file.write("--------------------------------\n")
    file.write(report)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.title("MNIST CNN Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

correct_indices = np.where(
    y_pred == y_test
)[0]

incorrect_indices = np.where(
    y_pred != y_test
)[0]

plt.figure(figsize=(12, 8))

for i, index in enumerate(correct_indices[:20]):

    plt.subplot(4, 5, i + 1)

    plt.imshow(
        x_test[index].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[index]}\n"
        f"Predicted: {y_pred[index]}"
    )

    plt.axis("off")

plt.suptitle("Correct Predictions")

plt.tight_layout()

correct_path = os.path.join(
    OUTPUT_DIR,
    "correct_predictions.png"
)

plt.savefig(
    correct_path,
    dpi=300
)

plt.close()

plt.figure(figsize=(12, 8))

for i, index in enumerate(incorrect_indices[:20]):

    plt.subplot(4, 5, i + 1)

    plt.imshow(
        x_test[index].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Actual: {y_test[index]}\n"
        f"Predicted: {y_pred[index]}"
    )

    plt.axis("off")

plt.suptitle("Incorrect Predictions")

plt.tight_layout()

incorrect_path = os.path.join(
    OUTPUT_DIR,
    "incorrect_predictions.png"
)

plt.savefig(
    incorrect_path,
    dpi=300
)

plt.close()

summary_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_summary.txt"
)

with open(summary_path, "w") as file:

    file.write("MNIST CNN Evaluation Summary\n")
    file.write("============================\n\n")

    file.write(f"Total Test Images: {len(y_test)}\n")
    file.write(f"Correct Predictions: {len(correct_indices)}\n")
    file.write(f"Incorrect Predictions: {len(incorrect_indices)}\n\n")

    file.write(f"Weighted Precision: {precision:.4f}\n")
    file.write(f"Weighted Recall:    {recall:.4f}\n")
    file.write(f"Weighted F1-Score:  {f1:.4f}\n")

print("\n===================================")
print("      EVALUATION COMPLETED")
print("===================================")

print(f"Classification report: {report_path}")
print(f"Confusion matrix:      {cm_path}")
print(f"Correct predictions:   {correct_path}")
print(f"Incorrect predictions: {incorrect_path}")
print(f"Evaluation summary:    {summary_path}")

print("\nAll evaluation files generated successfully.")