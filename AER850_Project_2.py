import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import utils, Sequential, layers, regularizers, callbacks, models
import matplotlib.pyplot as plt
from datetime import datetime


# Defining folder locations
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "data")
model_folder = os.path.join(script_dir, "models")

test_folder = os.path.join(data_folder, "test")
train_folder = os.path.join(data_folder, "train")
valid_folder = os.path.join(data_folder, "valid")


# Loading images and preprocessing
image_shape = (128,128,1)

def preprocess_image(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

test_images = utils.image_dataset_from_directory(
    directory = test_folder,
    labels = "inferred",
    image_size = (image_shape[0], image_shape[1])
    ).map(preprocess_image)

train_images = utils.image_dataset_from_directory(
    directory = train_folder,
    labels = "inferred",
    image_size = (image_shape[0], image_shape[1])
    ).map(preprocess_image)

valid_images = utils.image_dataset_from_directory(
    directory = valid_folder,
    labels = "inferred",
    image_size = (image_shape[0], image_shape[1])
    ).map(preprocess_image)


# Creating model
model = Sequential(
    [
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        
        layers.Dense(96, activation="leaky_relu", kernel_regularizer=regularizers.L1L2()),
        layers.Dropout(0.3),
        layers.Dense(96, activation="leaky_relu", kernel_regularizer=regularizers.L1L2()),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax", kernel_regularizer=regularizers.L1L2())
     ]
    )

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
    )

early_stopping = callbacks.EarlyStopping(
    monitor = "val_accuracy",  
    patience = 10,  
    restore_best_weights = True  
    )

history = model.fit(
    train_images,
    epochs = 50,
    validation_data = valid_images,
    callbacks = [early_stopping]
    )


# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


# Save model
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = f"model_{timestamp}.keras"

model_path = os.path.join(model_folder, model_filename)
model.save(model_path)

