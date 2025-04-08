# -*- coding: utf-8 -*-
# @Author: Daniel Fernandes
# @Date:   2025-04-08 15:19:09
# @Last Modified by:   Daniel Fernandes
# @Last Modified time: 2025-04-08 16:20:35
"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2025/04/06
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
Accelerator: GPU
"""

import os
import dvc.api
import numpy as np
import cv2
from keras import layers, Sequential, utils
from keras.models import Model
from dvclive import Live
from dvclive.keras import DVCLiveCallback

# Constants
DATA_FOLDER = "data"
TRAIN_FOLDER = os.path.join(DATA_FOLDER, "train")
TEST_FOLDER = os.path.join(DATA_FOLDER, "validation")
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 15


def load_data(folder, percentage=100):
    """
    Load a percentage of images and labels from a given folder.

    Args:
        folder (str): Path to the folder containing labeled subfolders.
        percentage (int): Percentage of data to load (1-100).

    Returns:
        tuple: Numpy arrays of images and labels.
    """
    images, labels = [], []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        files = os.listdir(label_folder)
        num_files = int(len(files) * (percentage / 100))
        for file in files[:num_files]:
            img_path = os.path.join(label_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))  # Ensure images are 28x28
            images.append(img)
            labels.append(int(label))
    return np.array(images), np.array(labels)


def preprocess_data(x, y):
    """
    Preprocess image data and labels.

    Args:
        x (np.array): Image data.
        y (np.array): Labels.

    Returns:
        tuple: Preprocessed image data and one-hot encoded labels.
    """
    x = x.astype("float32") / 255  # Scale images to [0, 1]
    x = np.expand_dims(x, -1)  # Ensure images have shape (28, 28, 1)
    y = utils.to_categorical(y, NUM_CLASSES)  # One-hot encode labels
    return x, y


def build_model(input_shape, num_classes):
    """
    Build a simple convolutional neural network model.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def main():
    """
    Main function to load data, build the model, train, and evaluate.
    """

    with Live() as live:

        params = dvc.api.params_show()
        percentage = params['ds_percentage']
        # Load and preprocess data
        print('Loading', percentage ,'% of the training data...')
        x_train, y_train = load_data(TRAIN_FOLDER, percentage)
        print('Loading', percentage, '%validation data...')
        x_test, y_test = load_data(TEST_FOLDER, percentage)
        print('Preprocess training data...')
        x_train, y_train = preprocess_data(x_train, y_train)
        print('Preprocess validation data...')
        x_test, y_test = preprocess_data(x_test, y_test)

        print("x_train shape:", x_train.shape)
        print(f"{x_train.shape[0]} train samples")
        print(f"{x_test.shape[0]} test samples")

        # Build and train the model
        model = build_model(INPUT_SHAPE, NUM_CLASSES)
        model.summary()

        print('Starting training...')
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.3,
            verbose=1,
            callbacks=[DVCLiveCallback(live)]
        )
        print('Training phase is done!')

        # Evaluate the model
        print('Starting evaluation...')
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        live.log_metric(f"test_loss", score[0], plot=False)
        live.log_metric(f"test_acc", score[1], plot=False)

        model.save("mnist.keras")
        live.log_artifact("mnist.keras",name="mnist.keras")



if __name__ == "__main__":
    main()