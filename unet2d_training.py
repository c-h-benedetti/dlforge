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
import tensorflow_datasets as tfds

from tqdm import tqdm
import tifffile
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              SETTINGS                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 1. SETTINGS

#@markdown ## 📍 a. Data paths

data_folder           = "/home/benedetti/Desktop/eaudissect/output_folder/"       #@param {type: "string"}
qc_folder             = None                                                      #@param {type: "string"}
inputs_name           = "images"                                                  #@param {type: "string"}
masks_name            = "outlines"                                                   #@param {type: "string"}
models_path           = "/home/benedetti/Desktop/eaudissect/output_folder/models" #@param {type: "string"}
working_directory     = "/home/benedetti/Desktop/eaudissect"                      #@param {type: "string"}
model_name_prefix     = "UNet2D"                                                  #@param {type: "string"}
name_suffix           = ".tif"                                                    #@param {type: "string"}

#@markdown ## 📍 b. Network architecture

validation_percentage = 0.15   #@param {type: "slider", min: 0.05, max: 0.95, step:0.05}
batch_size            = 16     #@param {type: "integer"}
epochs                = 50     #@param {type: "integer"}
unet_depth            = 4      #@param {type: "integer"}
num_filters_start     = 16     #@param {type: "integer"}
dropout_rate          = 0.25   #@param {type: "slider", min: 0.0, max: 0.5, step: 0.05}
optimizer             = 'Adam' #@param ["Adam", "SGD", "RMSprop"]
learning_rate         = 0.001  #@param {type: "number"}

#@markdown ## 📍 c. Data augmentation

use_data_augmentation = True  #@param {type: "boolean"}
augmentation_factor   = 2     #@param {type: "slider", min: 2, max: 6}
use_mirroring         = True  #@param {type: "boolean"}
use_gaussian_noise    = True  #@param {type: "boolean"}
rotation_90           = True  #@param {type: "boolean"}
use_gamma_correction  = True  #@param {type: "boolean"}
gamma_range           = 0.6   #@param {type: "slider", min:0.1, max:1.0}



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            SANITY CHECK                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 2. SANITY CHECK

#@markdown ## 📍 a. Data check

def is_folder_clean(root_folder):
    """
    The folders should only contain the training data, nothing else.

    Args:
        root_folder (str): The root folder containing the images and masks folders

    Returns:
        bool: True if the folders are clean, False otherwise.
    """
    images_path = os.path.join(root_folder, inputs_name)
    masks_path = os.path.join(root_folder, masks_name)
    imgs_list = set([f for f in os.listdir(images_path)])
    masks_list = set([f for f in os.listdir(masks_path)])
    imgs_set = set([i for i in imgs_list if i.endswith(name_suffix)])
    masks_set = set([i for i in masks_list if i.endswith(name_suffix)])
    if len(imgs_list) != len(imgs_set):
        print(f"❗ Images folder contains wrong data: {imgs_list.difference(imgs_set)}")
        return False
    if len(masks_list) != len(masks_set):
        print(f"❗ Masks folder contains wrong data: {masks_list.difference(masks_set)}")
        return False
    return True

def is_data_size_identical(root_folder):
    """
    All images must be the same size.
    The mask must be the same size as the input.

    Args:
        root_folder (str): The root folder containing the images and masks folders

    Returns:
        bool: True if the data is consistent, False otherwise.
    """
    images_path = os.path.join(root_folder, inputs_name)
    masks_path = os.path.join(root_folder, masks_name)
    imgs_list = [f for f in os.listdir(images_path) if f.endswith(name_suffix)]
    masks = [f for f in os.listdir(masks_path) if f.endswith(name_suffix)]
    ref_size = None
    src_name = None
    for img, mask in zip(imgs_list, masks):
        img_path = os.path.join(images_path, img)
        mask_path = os.path.join(masks_path, mask)
        img_data = tifffile.imread(img_path)
        mask_data = tifffile.imread(mask_path)
        if ref_size is None:
            ref_size = img_data.shape
            src_name = img
        if img_data.shape != ref_size:
            print(f"❗ All images don't have the same size: {img_data.shape} in {img} VS {ref_size} in {src_name}.")
            return False
        if mask_data.shape != img_data.shape:
            print(f"❗ A mask has a different size than its associated input: {img_data.shape} VS {mask_data.shape}.")
            return False
    return True

def is_all_data_useful(root_folder):
    """
    There must not be empty masks or empty images.

    Args:
        root_folder (str): The root folder containing the images and masks folders

    Returns:
        bool: True if the data is consistent, False otherwise.
    """
    images_path = os.path.join(root_folder, inputs_name)
    masks_path = os.path.join(root_folder, masks_name)
    imgs_list = [f for f in os.listdir(images_path) if f.endswith(name_suffix)]
    masks = [f for f in os.listdir(masks_path) if f.endswith(name_suffix)]
    for img, mask in zip(imgs_list, masks):
        img_path = os.path.join(images_path, img)
        mask_path = os.path.join(masks_path, mask)
        img_data = tifffile.imread(img_path)
        mask_data = tifffile.imread(mask_path)
        if len(np.unique(img_data)) == 1:
            print(f"❗ Image {img} is empty.")
            return False
        if len(np.unique(mask_data)) == 1:
            print(f"❗ Mask {mask} is empty.")
            return False
    return True

def is_content_identical(root_folder):
    """
    There must be the same number of images in both folders.
    There must be at least 15 images to start a training.

    Args:
        root_folder (str): The root folder containing the images and masks folders

    Returns:
        bool: True if the data is consistent, False otherwise.
    """
    images_path = os.path.join(root_folder, inputs_name)
    masks_path  = os.path.join(root_folder, masks_name)
    imgs_list   = set([f for f in os.listdir(images_path) if f.endswith(name_suffix)])
    masks_list  = set([f for f in os.listdir(masks_path) if f.endswith(name_suffix)])
    itr         = imgs_list.intersection(masks_list)
    if (len(itr) != len(masks_list)) or (len(itr) != len(imgs_list)):
        print(f"❗ The content of the images and masks folders is not identical: {imgs_list.difference(masks_list)}")
        return False
    if len(itr) < 15:
        print("❗ Not enough images to start a training")
        return False
    return True

#@markdown ## 📍 b. Checking runner

# List of sanity checks to perform before starting the training.
_SANITY_CHECK = [
    is_data_size_identical,
    is_all_data_useful,
    is_content_identical,
    is_folder_clean
]

def sanity_check(root_folder):
    for check in _SANITY_CHECK:
        if not check(root_folder):
            raise ValueError(f"💩 You should take a second look at your data... ({root_folder})")
    print("🎉 The data looks alright!")
    return True

#@markdown ## 📍 c. Running sanity check

sanity_check(data_folder)
if qc_folder is not None:
    sanity_check(qc_folder)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            DATA AUGMENTATION                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 3. DATA AUGMENTATION

#@markdown ## 📍 a. Utils

def probe_input_shape():
    """
    Searches for the first image in the images folder to determine the input shape of the model.

    Returns:
        tuple: The shape of the input image.
    """
    images_path = os.path.join(data_folder, inputs_name)
    input_shape = None
    content = [f for f in os.listdir(images_path) if f.endswith(name_suffix)]
    if len(content) == 0:
        return None
    img = tifffile.imread(os.path.join(images_path, content[0]))
    if len(img.shape) == 2:
        input_shape = img.shape + (1,)
    elif len(img.shape) == 3:
        input_shape = img.shape
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return input_shape

#@markdown ## 📍 b. Data augmentation functions

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
    gamma = tf.random.uniform(shape=[], minval=1.0 - gamma_range, maxval=1.0 + gamma_range)
    return tf.image.adjust_gamma(image, gamma=gamma)

def normalize(image):
    """
    Normalizes the image to have values between 0 and 1.

    Args:
        image (tf.Tensor): The input image.
    
    Returns:
        tf.Tensor: The normalized image.
    """
    m = np.min(image)
    M = np.max(image)
    return (image - m) / (M - m)

#@markdown ## 📍 c. Data augmentation layer

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
    input_shape = probe_input_shape()
    if use_data_augmentation:
        if use_mirroring:
            pipeline.append(RandomFlip(mode='horizontal_and_vertical'))
        if use_gaussian_noise:
            pipeline.append(GaussianNoise(0.02))
        if rotation_90:
            pipeline.append(Lambda(rotation_90_step, output_shape=input_shape))
        if use_gamma_correction:
            pipeline.append(Lambda(gamma_correction, output_shape=input_shape))
    pipeline.append(Lambda(normalize, output_shape=input_shape))
    return Sequential(pipeline)

def visualize_augmentations(images_path, augmentation_layer, num_examples=5):
    """
    Creates a very basic model composed of an input layer and the augmentation layer.
    Feeds the input image to the model to generate augmented images.
    Displays a few examples of augmented images, from the same source.
    """
    image_name = random.choice(os.listdir(images_path))
    image_path = os.path.join(images_path, image_name)
    # Load and preprocess the image
    image = tifffile.imread(image_path)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    
    # Create a temporary model
    inputs = Input(shape=image.shape[1:])
    outputs = augmentation_layer(inputs)
    temp_model = Model(inputs, outputs)
    
    # Generate augmented images
    _, axes = plt.subplots(1, num_examples, figsize=(20, 20))
    for i in range(num_examples):
        augmented_image = temp_model(image, training=True)  # training=True to apply augmentation
        axes[i].imshow(tf.squeeze(augmented_image).numpy())
        axes[i].axis('off')
    plt.show()

#@markdown ## 📍 d. Preview augmented data

images_path = os.path.join(data_folder, inputs_name)
visualize_augmentations(images_path, generate_data_augment_layer(), 10)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            MODEL GENERATOR                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 4. MODEL GENERATOR

#@markdown ## 📍 a. Utils

def get_version():
    """
    Used to auto-increment the version number of the model.
    Since each model is saved in a separate folder, we need to find the latest version number.
    Starts at 1 when the destination folder is empty.

    Returns:
        int: The next version number, that doesn't exist yet in the models folder.
    """
    if not os.path.isdir(models_path):
        os.makedirs(models_path)
    content = sorted([f for f in os.listdir(models_path) if f.startswith(model_name_prefix) and os.path.isdir(os.path.join(models_path, f))])
    if len(content) == 0:
        return 1
    else:
        return int(content[-1].split('-')[-1].replace('V', '')) + 1

#@markdown ## 📍 b. UNet2D architecture

def create_unet2d_model(input_shape):
    """
    Generates a UNet2D model.
    The model is created according to the images found in the input folder.
    The depth of the auto-encoder is defined by the unet_depth parameter.
    """
    inputs = Input(shape=input_shape)
    x = generate_data_augment_layer()(inputs)

    # Encoder:
    skip_connections = []
    for i in range(unet_depth):
        num_filters = num_filters_start * 2**i
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = Conv2D(num_filters, 3, padding='same')(x)
        skip_connections.append(x)
        x = MaxPooling2D(2)(x)
        x = Dropout(dropout_rate)(x)
    # Decoder:
    for i in reversed(range(unet_depth)):
        num_filters = num_filters_start * 2**i
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_connections[i]])
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def make_test_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        
        # Utiliser UpSampling pour restaurer la taille des images
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        
        # Couche de sortie pour prédiction du masque
        layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')
    ])
    
    return model

#@markdown ## 📍 c. Alternative loss functions

def jaccard_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return 1 - (intersection + 1) / (union + 1)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = - alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t + K.epsilon())
        return tf.reduce_mean(fl)
    return focal_loss_fixed

def tversky_loss(alpha=0.5, beta=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)
        return 1 - (true_pos + 1) / (true_pos + alpha * false_neg + beta * false_pos + 1)
    return loss

#@markdown ## 📍 d. Model instanciator

def instanciate_model():
    input_shape = probe_input_shape()
    model = create_unet2d_model(input_shape)
    model.compile(
        optimizer=optimizer, 
        loss=dice_loss, 
        metrics=[
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Accuracy()
        ]
    )
    return model

#@markdown ## 📍 e. Instanciate the model

model = instanciate_model()
model.summary()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            DATA MIGRATION                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 4. DATA MIGRATION

#@markdown ## 📍 a. Utils

_LOCAL_FOLDERS = ["training", "validation", "testing"]

def create_local_dirs():
    if not os.path.isdir(working_directory):
        raise ValueError(f"Working directory '{working_directory}' does not exist.")
    leaves = [inputs_name, masks_name]
    for r in _LOCAL_FOLDERS:
        for l in leaves:
            path = os.path.join(working_directory, r, l)
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path)

def copy_to(src_folder, dst_folder, files):
    for f in files:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f)
        shutil.copy(src_path, dst_path)

def migrate_data():
    images_path = os.path.join(data_folder, inputs_name)
    masks_path = os.path.join(data_folder, masks_name)
    qc_input_path = None if qc_folder is None else os.path.join(qc_folder, inputs_name)
    qc_masks_path = None if qc_folder is None else os.path.join(qc_folder, masks_name)
    # Creating a validation set.
    content = [f for f in os.listdir(images_path) if f.endswith(name_suffix)]
    random.shuffle(content)
    last_idx = int(validation_percentage * len(content))

    copy_to(images_path, os.path.join(working_directory, "validation", inputs_name), content[:last_idx])
    copy_to(masks_path, os.path.join(working_directory, "validation", masks_name), content[:last_idx])
    copy_to(images_path, os.path.join(working_directory, "training", inputs_name), content[last_idx:])
    copy_to(masks_path, os.path.join(working_directory, "training", masks_name), content[last_idx:])

    if (qc_input_path is not None) and (qc_masks_path is not None):
        qc_content = [f for f in os.listdir(qc_input_path) if f.endswith(name_suffix)]
        copy_to(qc_input_path, os.path.join(working_directory, "testing", inputs_name), qc_content)
        copy_to(qc_masks_path, os.path.join(working_directory, "testing", masks_name), qc_content)

def get_local_paths(source):
    img_dst_val  = os.path.join(working_directory, source, inputs_name)
    mask_dst_val = os.path.join(working_directory, source, masks_name)
    return {
        'images': [os.path.join(img_dst_val, f) for f in os.listdir(img_dst_val)],
        'masks' : [os.path.join(mask_dst_val, f) for f in os.listdir(mask_dst_val)]
    }

#@markdown ## 📍 b. Datasets generator

def load_dataset(root_folder):
    images = []
    masks = []
    content = sorted([f for f in os.listdir(os.path.join(root_folder, inputs_name)) if f.endswith('.tif')])
    for c in content:
        img = tifffile.imread(os.path.join(root_folder, inputs_name, c))
        mask = tifffile.imread(os.path.join(root_folder, masks_name, c))
        img = np.expand_dims(img, axis=-1) 
        mask = np.expand_dims(mask, axis=-1)
        images.append(img)
        masks.append(mask)
    images = np.array(images)
    masks = np.array(masks)
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    ds = ds.batch(batch_size).shuffle(buffer_size=100).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


#@markdown ## 📍 c. Migration and dataset creation

create_local_dirs()
migrate_data()

training_dataset   = load_dataset(os.path.join(working_directory, "training"))
validation_dataset = load_dataset(os.path.join(working_directory, "validation"))
testing_dataset    = None

if qc_folder is not None:
    testing_dataset = load_dataset(os.path.join(working_directory, "testing"))
else:
    print("😱 No quality control will be performed.")



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            TRAINING THE MODEL                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 5. TRAINING THE MODEL

#@markdown ## 📍 a. Training launcher

def train_model(model, train_dataset, val_dataset):
    # Path of the folder in which the model is exported.
    v = get_version()
    version_name = f"{model_name_prefix}-V{str(v).zfill(3)}"
    output_path = os.path.join(models_path, version_name)
    os.makedirs(output_path)

    print(f"💾 Exporting model to: {output_path}")

    checkpoint = ModelCheckpoint('best.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, mode='min')

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=100,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    model.save(os.path.join(output_path, 'last.keras'))
    return history

#@markdown ## 📍 b. Training the model

history = train_model(model, training_dataset, validation_dataset)