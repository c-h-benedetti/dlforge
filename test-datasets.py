import os
import numpy as np
import tifffile
from random import shuffle
import random
import matplotlib.pyplot as plt

from unet2d_training import (get_shape, get_data_sets)
from unet2d_training import (working_directory, inputs_name, masks_name)
from unet2d_training import (rotation_range_2_pi, gamma_range)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, RandomRotation,
                                     UpSampling2D, concatenate, Dropout,
                                     RandomFlip, GaussianNoise, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


"""
DATASETS:

Functions responsible for creating datasets for training and testing.
They can generate pairs (image + mask) or individual files for testing.
They are based on generators, so they can handle data that does not fit in memory.
"""


def open_pair(input_path, mask_path, img_only):
    raw = tifffile.imread(input_path)
    raw = np.expand_dims(raw, axis=-1)
    image = tf.constant(raw, dtype=tf.float32)
    raw = tifffile.imread(mask_path)
    raw = np.expand_dims(raw, axis=-1)
    mask = tf.constant(raw, dtype=tf.float32)
    if img_only:
        return image
    else:
        return (image, mask)

def pairs_generator(src, img_only):
    source = src.decode('utf-8')
    _, l_files = get_data_sets(os.path.join(working_directory, source), [inputs_name], True)
    l_files = sorted(list(l_files))
    i = 0
    while i < len(l_files):
        input_path = os.path.join(working_directory, source, inputs_name, l_files[i])
        mask_path = os.path.join(working_directory, source, masks_name, l_files[i])
        yield open_pair(input_path, mask_path, img_only)
        i += 1

def make_dataset(source, img_only=False):
    shape = get_shape()
    
    output_signature=tf.TensorSpec(shape=shape, dtype=tf.float32, name=None)
    if not img_only:
        output_signature = (output_signature, tf.TensorSpec(shape=shape, dtype=tf.uint8, name=None))
    
    ds = tf.data.Dataset.from_generator(
        pairs_generator,
        args=(source, img_only),
        output_signature=output_signature
    )
    return ds

def test_ds_consumer():
    batch = 20 # will be equivalent to the batch size
    take = 10 # will be equivalent to the number of epochs
    
    ds_counter = make_dataset("training")
    for i, (image, mask) in enumerate(ds_counter.repeat().batch(batch).take(take)):
        print(f"{str(i+1).zfill(2)}: ", image.shape, mask.shape)

    print("\n================\n")

    ds_counter = make_dataset("training", True)
    for i, image in enumerate(ds_counter.repeat().batch(batch).take(take)):
        print(f"{str(i+1).zfill(2)}: ", image.shape)
    
    print("\nDONE.")


"""
DATA AUGMENTATION:
"""

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
    pipeline.append(RandomRotation(rotation_range_2_pi/100, fill_mode='reflect'))
    pipeline.append(Lambda(gamma_correction, output_shape=input_shape))
    pipeline.append(Lambda(normalize, output_shape=input_shape))
    return Sequential(pipeline)

def visualize_augmentations(augmentation_layer, num_examples=5, one_shot=True):
    s = get_shape()  # Assuming this returns the shape (e.g., (128, 128, 1))
    ds = make_dataset("training", True).batch(1).take(num_examples)
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


################################################################################

if __name__ == '__main__':
    # test_ds_consumer()
    dal = generate_data_augment_layer()
    visualize_augmentations(dal, num_examples=5)