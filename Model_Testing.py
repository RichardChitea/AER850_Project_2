import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import utils, models, preprocessing
import matplotlib.pyplot as plt
import cv2


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
model_path = os.path.join(model_folder,"model_2024-11-17_09-28-50.keras")

model = models.load_model(model_path, safe_mode=False)
test_loss, test_acc = model.evaluate(x=test_images)


# Load and process the three test images
img_paths = [
    os.path.join(data_folder, "test", "crack", "test_crack.jpg"),
    os.path.join(data_folder, "test", "missing-head", "test_missinghead.jpg"),
    os.path.join(data_folder, "test", "paint-off", "test_paintoff.jpg")]

img_arrays = []
for img_path in img_paths:
    img = preprocessing.image.load_img(img_path, 
                                       target_size=(image_shape[0], 
                                       image_shape[1]),
                                       color_mode="grayscale")
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_arrays.append(img_array)

img_batch = np.vstack(img_arrays)


# Predict and display
predictions = model.predict(img_batch)

class_labels = ['Crack', 'Missing Head', 'Paint-off']

for i, img_path in enumerate(img_paths):
    img_display = cv2.imread(img_path)
    img_display = cv2.resize(img_display, (500, 500))

    for j, (label, probability) in enumerate(zip(class_labels, predictions[i])):
        text = f"{label}:{probability * 100:.1f}%"
        position = (10, 425 + j*30)
        cv2.putText(img_display, text, position, cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 122, 0), 2, cv2.LINE_8)

    plt.figure(figsize = (6, 6))
    plt.imshow(img_display)
    plt.axis('off')
    plt.title(f"True Crack Classification Label: {class_labels[i]}\nPredicted Crack Classification Label: {class_labels[np.argmax(predictions[i])]}")
    plt.show()






