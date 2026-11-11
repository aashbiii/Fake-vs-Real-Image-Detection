#!/usr/bin/env python3
"""
Fake vs Real Image Detection using Transfer Learning (EfficientNet + TensorFlow/Keras)
Author: Ashbi Saju (MCA S2 — Artificial Intelligence using Gen AI)

This script trains, evaluates, and runs inference for detecting synthetic AI-generated images
(Stable Diffusion / GANs) versus authentic real photographs using Transfer Learning.
"""

import os
import argparse
import warnings
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics import classification_report, confusion_matrix


def get_data_augmentation():
    """Builds real-time data augmentation pipeline."""
    return tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.15),
        layers.RandomTranslation(0.1, 0.1)
    ], name="data_augmentation")


def build_model(img_size=(300, 300), backbone="EfficientNetB3", trainable_base=False):
    """
    Builds transfer learning classifier with pretrained EfficientNet feature extractor.
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="input_image")
    
    aug_pipeline = get_data_augmentation()
    x = aug_pipeline(inputs)

    if backbone == "EfficientNetB0":
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size[0], img_size[1], 3)
        )
    else:
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size[0], img_size[1], 3)
        )

    base_model.trainable = trainable_base

    x = base_model(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation='relu', name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_classification")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FakeRealDetector")
    return model, base_model


def load_datasets(data_dir, img_size=(300, 300), batch_size=16, seed=42):
    """
    Loads training and test datasets from directory partitions.
    Expected structure:
      data_dir/train/REAL, data_dir/train/FAKE
      data_dir/test/REAL,  data_dir/test/FAKE
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing train or test directory inside '{data_dir}'")

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        label_mode='binary',
        shuffle=True
    )

    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        label_mode='binary',
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Loaded classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, test_ds, class_names


def train_pipeline(data_dir, img_size=(300, 300), batch_size=16, epochs_p1=10, epochs_p2=15,
                   backbone="EfficientNetB3", save_path="fake_real_detector.h5"):
    """
    Executes two-phase transfer learning training strategy.
    Phase 1: Feature Extraction (base frozen, Adam lr=1e-3)
    Phase 2: Fine-Tuning (top layers unfrozen, Adam lr=1e-5)
    """
    print("=" * 60)
    print("1. Loading CIFAKE Dataset...")
    print("=" * 60)
    train_ds, test_ds, class_names = load_datasets(data_dir, img_size, batch_size)

    print("\n" + "=" * 60)
    print(f"2. Building Model with {backbone} Backbone...")
    print("=" * 60)
    model, base_model = build_model(img_size=img_size, backbone=backbone, trainable_base=False)
    model.summary()

    # --- Phase 1: Feature Extraction ---
    print("\n" + "=" * 60)
    print("3. Phase 1: Feature Extraction (Classification Head Training)...")
    print("=" * 60)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    cb_early = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    cb_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    history1 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs_p1,
        callbacks=[cb_early, cb_lr]
    )

    # --- Phase 2: Fine Tuning ---
    print("\n" + "=" * 60)
    print("4. Phase 2: Fine-Tuning Top Convolutional Layers...")
    print("=" * 60)
    base_model.trainable = True
    # Freeze bottom layers and unfreeze top 60 layers
    for layer in base_model.layers[:-60]:
        layer.trainable = False

    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Fine-tuning active. Trainable parameters: {trainable_count:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    cb_early2 = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    history2 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs_p2,
        callbacks=[cb_early2]
    )

    # --- Evaluation ---
    print("\n" + "=" * 60)
    print("5. Evaluating Model on Unseen Test Dataset...")
    print("=" * 60)
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nFinal Test Accuracy : {acc * 100:.2f}%")
    print(f"Final Test Loss     : {loss:.4f}\n")

    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten().astype(int)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save model
    model.save(save_path)
    print(f"Model successfully saved to {save_path}")

    # Plot metrics
    plot_training_curves(history1, history2, output_file="training_curves.png")
    plot_confusion_matrix(y_true, y_pred, class_names, output_file="confusion_matrix.png")

    return model


def plot_training_curves(h1, h2, output_file="training_curves.png"):
    """Plots and saves combined training and validation loss & accuracy curves."""
    acc = h1.history['accuracy'] + h2.history['accuracy']
    vacc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    vloss = h1.history['val_loss'] + h2.history['val_loss']
    split_epoch = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — Fake vs Real Image Detection', fontsize=14, fontweight='bold')

    ax1.plot(acc, label='Train Accuracy', color='steelblue', lw=2)
    ax1.plot(vacc, label='Val Accuracy', color='orange', lw=2)
    ax1.axvline(split_epoch, color='red', linestyle='--', label='Fine-tuning Started')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss, label='Train Loss', color='steelblue', lw=2)
    ax2.plot(vloss, label='Val Loss', color='orange', lw=2)
    ax2.axvline(split_epoch, color='red', linestyle='--', label='Fine-tuning Started')
    ax2.set_title('Model Loss (Binary Cross-Entropy)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved training history curves to '{output_file}'")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_file="confusion_matrix.png"):
    """Generates and saves the classification confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor='white')
    plt.title('Confusion Matrix — Fake vs Real', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved confusion matrix heatmap to '{output_file}'")
    plt.close()


def predict_single_image(model_path, image_path, img_size=(300, 300)):
    """Infers whether an input image is REAL or FAKE using a trained model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at '{image_path}'")

    print(f"Loading model from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    img = keras_image.load_img(image_path, target_size=img_size)
    img_array = keras_image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    pred_prob = model.predict(img_array, verbose=0)[0][0]
    
    # Standard class ordering: 0 -> FAKE, 1 -> REAL
    is_real = pred_prob > 0.5
    label = "REAL" if is_real else "FAKE"
    confidence = pred_prob if is_real else (1.0 - pred_prob)

    print("\n" + "=" * 40)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Probability: {pred_prob:.4f} (Threshold: 0.5)")
    print("=" * 40)

    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Fake vs Real Image Detection via Transfer Learning")
    subparsers = parser.add_subparsers(dest="command")

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train the model on CIFAKE dataset")
    train_parser.add_argument("--data_dir", type=str, default="cifake", help="Path to CIFAKE root folder")
    train_parser.add_argument("--img_size", type=int, default=300, help="Image input dimension (square)")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    train_parser.add_argument("--epochs_p1", type=int, default=10, help="Phase 1 feature extraction epochs")
    train_parser.add_argument("--epochs_p2", type=int, default=15, help="Phase 2 fine-tuning epochs")
    train_parser.add_argument("--backbone", type=str, default="EfficientNetB3", choices=["EfficientNetB0", "EfficientNetB3"])
    train_parser.add_argument("--save_path", type=str, default="fake_real_detector.h5", help="Saved model file path")

    # Predict subparser
    pred_parser = subparsers.add_parser("predict", help="Predict single image authenticity")
    pred_parser.add_argument("--image", type=str, required=True, help="Path to target image")
    pred_parser.add_argument("--model", type=str, default="fake_real_detector.h5", help="Path to trained model .h5")
    pred_parser.add_argument("--img_size", type=int, default=300, help="Image input dimension (square)")

    args = parser.parse_args()

    if args.command == "train":
        train_pipeline(
            data_dir=args.data_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            epochs_p1=args.epochs_p1,
            epochs_p2=args.epochs_p2,
            backbone=args.backbone,
            save_path=args.save_path
        )
    elif args.command == "predict":
        predict_single_image(
            model_path=args.model,
            image_path=args.image,
            img_size=(args.img_size, args.img_size)
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
