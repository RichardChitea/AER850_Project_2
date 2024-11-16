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
    image_size = (image_shape[0], image_shape[1]),
    shuffle = False
    ).map(preprocess_image)


# Test model
model_path = os.path.join(model_folder,"model_2024-11-16_11-38-16.keras")

model = models.load_model(model_path, safe_mode=False)
test_loss, test_acc = model.evaluate(x=test_images)




