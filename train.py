import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

EPOCHS = 15
BATCH_SIZE = 128
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training images: {x_train.shape}")
print(f"Testing images:  {x_test.shape}")

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"Processed training shape: {x_train.shape}")
print(f"Processed testing shape:  {x_test.shape}")

model = models.Sequential([
    
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Architecture:")
model.summary()

checkpoint_path = os.path.join(
    MODEL_DIR,
    "mnist_cnn_best.keras"
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True
    )
]

print("\nStarting CNN training...\n")

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\n===================================")
print("        FINAL TEST RESULTS")
print("===================================")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("===================================\n")

final_model_path = os.path.join(
    MODEL_DIR,
    "mnist_cnn_final.keras"
)

model.save(final_model_path)

print(f"Model saved to: {final_model_path}")

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)

plt.savefig(accuracy_path, dpi=300)
plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

plt.savefig(loss_path, dpi=300)
plt.close()

history_file = os.path.join(
    OUTPUT_DIR,
    "training_history.txt"
)

with open(history_file, "w") as file:

    file.write("MNIST CNN Training Results\n")
    file.write("==========================\n\n")

    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n"
    )

    file.write("Training Accuracy:\n")

    for epoch, accuracy in enumerate(
        history.history["accuracy"],
        start=1
    ):
        file.write(
            f"Epoch {epoch}: {accuracy:.4f}\n"
        )

    file.write("\nValidation Accuracy:\n")

    for epoch, accuracy in enumerate(
        history.history["val_accuracy"],
        start=1
    ):
        file.write(
            f"Epoch {epoch}: {accuracy:.4f}\n"
        )


print("\nTraining completed successfully.")
print("Generated files:")
print(f"  - {final_model_path}")
print(f"  - {accuracy_path}")
print(f"  - {loss_path}")
print(f"  - {history_file}")