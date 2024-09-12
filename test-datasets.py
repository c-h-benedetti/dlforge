import os
import numpy as np
import tifffile
from random import shuffle
import random
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
                                     UpSampling2D, concatenate, Dropout,
                                     RandomFlip, GaussianNoise, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


# Images location
sources_path = "/home/benedetti/Desktop/eaudissect/training/"
inputs       = "images"
masks        = "outlines"

def get_files_list():
    l_files = [f for f in os.listdir(os.path.join(sources_path, inputs)) if f.endswith('.tif')]
    shuffle(l_files)
    return l_files

def get_shape():
    l_files = get_files_list()
    input_path = os.path.join(sources_path, inputs, l_files[0])
    raw = tifffile.imread(input_path)
    s = raw.shape
    if len(s) == 2:
        s = (s[0], s[1], 1)
    return s

def open_pair(input_path, mask_path):
    raw = tifffile.imread(input_path)
    raw = np.expand_dims(raw, axis=-1)
    image = tf.constant(raw, dtype=tf.float32)
    raw = tifffile.imread(mask_path)
    raw = np.expand_dims(raw, axis=-1)
    mask = tf.constant(raw, dtype=tf.float32)
    return image, mask

def pairs_generator():
    l_files = get_files_list()
    i = 0
    while i < len(l_files):
        input_path = os.path.join(sources_path, inputs, l_files[i])
        mask_path = os.path.join(sources_path, masks, l_files[i])
        yield open_pair(input_path, mask_path)
        i += 1

def images_generator():
    l_files = get_files_list()
    i = 0
    while i < len(l_files):
        input_path = os.path.join(sources_path, inputs, l_files[i])
        raw = tifffile.imread(input_path)
        raw = np.expand_dims(raw, axis=-1)
        yield tf.constant(raw, dtype=tf.float32)
        i += 1

def rotation_90_step(image):
    """
    Applies a random rotation of 90, 180 or 270 degrees to the image.

    Args:
        image (tf.Tensor): The input image.
    
    Returns:
        tf.Tensor: The rotated image.
    """
    angles = [0, 90, 180, 270]
    angle = tf.random.shuffle(angles)[0]
    return tf.image.rot90(image, k=angle // 90)

def gamma_correction(image):
    """
    Applies a random γ-correction to the image.
    γ == 1.0 > no change

    Args:
        image (tf.Tensor): The input image.
    
    Returns:
        tf.Tensor: The corrected image.
    """
    gamma = tf.random.uniform(shape=[], minval=1.0 - 0.2, maxval=1.0 + 0.2)
    return tf.image.adjust_gamma(image, gamma=gamma)

def normalize(image):
    """
    Normalizes the image to have values between 0 and 1.

    Args:
        image (tf.Tensor): The input image.
    
    Returns:
        tf.Tensor: The normalized image.
    """
    m = tf.reduce_min(image)
    M = tf.reduce_max(image)
    return (image - m) / (M - m)

def make_dataset(img_only=False):
    raw_shape = get_shape()
    if img_only:
        shape = raw_shape
    else:
        shape = (2,) + raw_shape
    ds = tf.data.Dataset.from_generator(
        pairs_generator if (not img_only) else images_generator, 
        output_types=tf.float32, 
        output_shapes=shape
    )
    return ds

def generate_data_augment_layer():
    """
    Generates a data augmentation layer.
    It takes the form of a Sequential layer to be plugged after the input layer of the model.
    Augmentation layers are disabled at inference.
    If the data augmentation is disabled, the layer only normalizes the input.
    Order of transform: flip, noise, rotation, gamma correction, normalization.

    Returns:
        tf.keras.Sequential: The data augmentation layer.
    """
    pipeline = []
    input_shape = get_shape()
    pipeline.append(RandomFlip(mode='horizontal_and_vertical'))
    pipeline.append(GaussianNoise(0.02))
    pipeline.append(Lambda(rotation_90_step, output_shape=input_shape))
    pipeline.append(Lambda(gamma_correction, output_shape=input_shape))
    pipeline.append(Lambda(normalize, output_shape=input_shape))
    return Sequential(pipeline)

def visualize_augmentations(augmentation_layer, num_examples=5, one_shot=True):
    s = get_shape()  # Assuming this returns the shape (e.g., (128, 128, 1))
    ds = make_dataset(True).batch(1).take(num_examples)
    grid_size = (2, num_examples) 
    
    # Create a temporary model for the augmentation
    inputs = Input(shape=s)
    outputs = augmentation_layer(inputs)
    temp_model = Model(inputs, outputs)

    # Prepare the grid for displaying images (two rows: one for original, one for augmented)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 10))
    axes = axes.flatten()  # Flatten axes to easily index them

    # Loop through the dataset and display original and augmented images in pairs
    for i, img_batch in enumerate(ds):
        if i >= num_examples:
            break
        # Get the original image
        original_image = img_batch[0].numpy()  # Convert Tensor to NumPy array
        # Get the augmented image
        if one_shot:
            augmented_image = temp_model(img_batch, training=True)[0].numpy()
        else:
            augmented_image = temp_model.predict(img_batch)[0]
        # Display the original image on the first row
        axes[i].imshow(original_image[..., 0], cmap='gray')  # Assuming grayscale images
        axes[i].axis('off')  # Hide axis
        # Display the augmented image on the second row
        axes[i + num_examples].imshow(augmented_image[..., 0], cmap='viridis')  # Assuming augmented is grayscale
        axes[i + num_examples].axis('off')  # Hide axis

    plt.tight_layout()
    plt.show()

def test_ds_consumer():
    batch = 20 # will be equivalent to the batch size
    take = 10 # will be equivalent to the number of epochs
    
    ds_counter = make_dataset()
    for i, count_batch in enumerate(ds_counter.repeat().batch(batch).take(take)):
        print(f"{str(i+1).zfill(2)}: ", count_batch.shape, type(count_batch))


if __name__ == '__main__':
    # test_ds_consumer()
    dal = generate_data_augment_layer()
    visualize_augmentations(dal, num_examples=5)