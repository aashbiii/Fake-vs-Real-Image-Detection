#!/usr/bin/env python3
"""
Lightweight Inference Script for Fake vs Real Image Detection
Author: Ashbi Saju
"""

import sys
import os
import argparse
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image


def classify_image(image_path, model_path="fake_real_detector.h5", img_size=300):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    img = keras_image.load_img(image_path, target_size=(img_size, img_size))
    img_array = keras_image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    is_real = prediction > 0.5
    label = "REAL" if is_real else "FAKE"
    confidence = prediction if is_real else (1.0 - prediction)

    print("\n" + "=" * 45)
    print(f"Target Image : {os.path.basename(image_path)}")
    print(f"Result       : {label} ({'Authentic Photograph' if is_real else 'AI-Generated / Synthetic'})")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print(f"Raw Sigmoid  : {prediction:.4f}")
    print("=" * 45 + "\n")

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify image as Real or Fake AI-generated")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", "-m", type=str, default="fake_real_detector.h5", help="Path to trained model")
    parser.add_argument("--size", "-s", type=int, default=300, help="Image input dimension")
    args = parser.parse_args()

    classify_image(args.image, args.model, args.size)
