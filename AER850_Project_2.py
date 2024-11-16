import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
#from keras.preprocessing.image import image_dataset_from_directory
from keras import utils

#image_path = 'data'
#dataset = image_dataset_from_directory(image_path, labels='inferred', label_mode='categorical')

#model = Sequential()

script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "data")

image_shape = (128,128,1)

test_folder = os.path.join(data_folder, "test")
train_folder = os.path.join(data_folder, "train")
valid_folder = os.path.join(data_folder, "valid")

def preprocess_image(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

test_images = utils.image_dataset_from_directory(
    directory=test_folder,
    labels="inferred",
    image_size=(image_shape[0], image_shape[1]),
    ).map(preprocess_image)

train_images = utils.image_dataset_from_directory(
    directory=train_folder,
    labels="inferred",
    image_size=(image_shape[0], image_shape[1]),
    ).map(preprocess_image)

valid_images = utils.image_dataset_from_directory(
    directory=valid_folder,
    labels="inferred",
    image_size=(image_shape[0], image_shape[1]),
    ).map(preprocess_image)

