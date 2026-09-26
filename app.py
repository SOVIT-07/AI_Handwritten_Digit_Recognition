import os
import uuid

import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from PIL import Image, ImageOps

app = Flask(__name__)

MODEL_PATH = "models/mnist_cnn_final.keras"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully.")

def preprocess_image(image_path):

    image = Image.open(image_path).convert("L")

    image = image.resize((28, 28))

    image_array = np.array(image)

    # Convert white background / black digit
    # into MNIST-style black background / white digit.
    if np.mean(image_array) > 127:
        image = ImageOps.invert(image)

    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict_image(image_path):

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

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "error": "No image file uploaded."
        })

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No file selected."
        })

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp"
    }

    if extension not in allowed_extensions:

        return jsonify({
            "success": False,
            "error": "Please upload PNG, JPG, JPEG, or BMP."
        })

    filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:

        digit, confidence = predict_image(
            filepath
        )

        return jsonify({
            "success": True,
            "digit": digit,
            "confidence": round(
                confidence * 100,
                2
            )
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        })


if __name__ == "__main__":

    print("\n===================================")
    print(" AI Handwritten Digit Recognition")
    print("===================================")
    print("Open: http://localhost:5000")
    print("===================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )