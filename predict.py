import os
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps

MODEL_PATH = "models/mnist_cnn_final.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully.")

def preprocess_image(image_path):
    """
    Preprocess a handwritten digit image so that it
    resembles the centered MNIST format.
    """

    image = Image.open(image_path).convert("L")

    image_array = np.array(image)

    if np.mean(image_array) > 127:
        image_array = 255 - image_array

    mask = image_array > 30

    coords = np.argwhere(mask)

    if coords.size == 0:
        raise ValueError(
            "No handwritten digit detected in the image."
        )

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    digit = Image.fromarray(cropped)

    # Make the image square
    width, height = digit.size
    size = max(width, height)

    padding = 20

    square_size = size + padding * 2

    square = Image.new(
        "L",
        (square_size, square_size),
        0
    )

    x_offset = (
        square_size - width
    ) // 2

    y_offset = (
        square_size - height
    ) // 2

    square.paste(
        digit,
        (x_offset, y_offset)
    )

    digit = square.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    final_image.paste(
        digit,
        (4, 4)
    )

    final_array = np.array(
        final_image
    ).astype("float32") / 255.0

    # CNN input shape: (1, 28, 28, 1)
    final_array = np.expand_dims(
        final_array,
        axis=-1
    )

    final_array = np.expand_dims(
        final_array,
        axis=0
    )

    return final_array

def predict_digit(image_path):
    """
    Predict the handwritten digit from an image.
    """

    processed_image = preprocess_image(
        image_path
    )

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_digit = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_digit]
    )

    return predicted_digit, confidence

if __name__ == "__main__":

    image_path = input(
        "\nEnter the path of the digit image: "
    ).strip()

    if not os.path.exists(image_path):

        print(
            f"\nError: Image not found: {image_path}"
        )

        raise SystemExit(1)

    digit, confidence = predict_digit(
        image_path
    )

    print("\n===================================")
    print("       DIGIT PREDICTION")
    print("===================================")
    print(f"Predicted Digit: {digit}")
    print(f"Confidence:      {confidence * 100:.2f}%")
    print("===================================\n")