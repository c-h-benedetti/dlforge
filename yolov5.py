import os
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import shutil
import tifffile
import re

"""

# YOLOv5 (Object detection with PyTorch)
----------------------------------------

Before starting using this script, please make sure that:
    - You have some annotated images.
    - You have the required modules installed.
    - You cloned/downloaded the YOLOv5 repository (https://github.com/ultralytics/yolov5.git).

"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              SETTINGS                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

"""

- `data_folder`      : Folder in which we can find the images and annotations folders.
- `qc_folder`        : Folder in which we can find the quality control images and masks folders.
- `inputs_name`      : Name of the folder containing the input images (name of the folder in `data_folder` and `qc_folder`.).
- `annotations_name` : Name of the folder containing the annotations (name of the folder in `data_folder` and `qc_folder`.).
- `models_path`      : Folder in which the models will be saved. They will be saved as "{model_name_prefix}-V{version_number}".
- `working_directory`: Folder in which the training, validation and testing folders will be created.
- `model_name_prefix`: Prefix of the model name. Will be part of the folder name in `models_path`.
- `reset_local_data` : If True, the locally copied training, validation and testing folders will be re-imported.

- `yolov5_path`          : Path to the localy downloaded YOLOv5 repository.
- `validation_percentage`: Percentage of the data that will be used for validation. This data will be moved to the validation folder.
- `batch_size`           : Number of images per batch.
- `epochs`               : Number of epochs for the training.
- `classes_names`        : Names of the classes that we will try to detect.
- `optimizer`            : Optimizer used for the training.
- `learning_rate`        : Learning rate of the optimizer.
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              SETTINGS                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 1. SETTINGS

#@markdown ## 📍 a. Data paths

data_folder       = "/home/benedetti/Desktop/eaudissect/deep-bacs/train/"
qc_folder         = "/home/benedetti/Desktop/eaudissect/deep-bacs/qc_data/"
inputs_name       = "images"
annotations_name  = "labels"
models_path       = "/home/benedetti/Desktop/eaudissect/yolo_working/models"
working_directory = "/home/benedetti/Desktop/eaudissect/yolo_local"
model_name_prefix = "YOLOv5"
reset_local_data  = True
remove_wrong_data = True

#@markdown ## 📍 b. Network architecture

yolov5_path           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")
validation_percentage = 0.15
batch_size            = 120
epochs                = 50
classes_names         = ['Microglia']
optimizer             = 'Adam'
learning_rate         = 0.0001

#@markdown ## 📍 c. Constants

_IMAGES_REGEX = re.compile(r"(.+)\.(png|jpg)$")
_ANNOTATIONS_REGEX = re.compile(r"(.+)\.(txt)$")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            SANITY CHECK                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 2. SANITY CHECK

"""
List of all the points checked during the sanity check:
    - [X] Is the content of folders valid?
    - [X] Does every image have its corresponding annotation?
    - [X] Is each annotation correctly formatted?
    - [ ] Does every annotation have at least one bounding box?
"""

#@markdown ## 📍 a. Sanity check functions

def _is_all_data_valid(path, regex):
    all_files = set(os.listdir(path))
    valid_files = set([f for f in all_files if regex.match(f)])
    if all_files == valid_files:
        return True
    else:
        print(f"Invalid files found in {path}: {all_files - valid_files}")
        return False

def is_all_data_valid(source_folder):
    s1 = _is_all_data_valid(os.path.join(source_folder, inputs_name), _IMAGES_REGEX)
    s2 = _is_all_data_valid(os.path.join(source_folder, annotations_name), _ANNOTATIONS_REGEX)
    return (s1 and s2)

def is_data_matching(source_folder):
    all_images = set([f.split('.')[0] for f in os.listdir(os.path.join(source_folder, inputs_name))])
    all_annotations = set([f.split('.')[0] for f in os.listdir(os.path.join(source_folder, annotations_name))])
    if all_images == all_annotations:
        return True
    else:
        print(f"Images and annotations do not match: {all_images - all_annotations}")
        return False

def _are_annotations_valid(source_folder, file):
    pattern = (int, float, float, float, float)
    with open(os.path.join(source_folder, annotations_name, file), 'r') as f:
        lines = f.readlines()
        for l in lines:
            if len(l) == 0 or l.startswith('#'):
                continue
            pieces = l.split(' ')
            if len(pieces) != 5:
                return False
            for t in zip(pattern, pieces):
                try:
                    t[0](t[1])
                except:
                    return False
    return True

def are_annotations_valid(source_folder):
    all_files = os.listdir(os.path.join(source_folder, annotations_name))
    all_annotations = [f for f in all_files if _ANNOTATIONS_REGEX.match(f)]
    status = []
    for f in all_annotations:
        if not _are_annotations_valid(source_folder, f):
            status.append(f)
    if len(status) == 0:
        return True
    else:
        print(f"Invalid annotations found: {status}")
        return False

def _are_annotations_useful(source_folder, file):
    count = 0
    with open(os.path.join(source_folder, annotations_name, file), 'r') as f:
        lines = f.readlines()
        for l in lines:
            if len(l) == 0 or l.startswith('#'):
                continue
            count += 1
    return count > 0

def are_all_annotations_useful(source_folder):
    all_files = os.listdir(os.path.join(source_folder, annotations_name))
    all_annotations = [f for f in all_files if _ANNOTATIONS_REGEX.match(f)]
    status = []
    for f in all_annotations:
        if not _are_annotations_useful(source_folder, f):
            status.append(f)
    if len(status) == 0:
        return True
    else:
        print(f"Empty annotations found: {status}")
        return False

#@markdown ## 📍 b. Launch sanity check

_SANITY_CHECK = [
    is_all_data_valid,
    is_data_matching,
    are_annotations_valid,
    are_all_annotations_useful
]

def sanity_check(source_folder):
    status = [f(source_folder) for f in _SANITY_CHECK]
    return all(status)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            DATA MIGRATION                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 3. DATA MIGRATION

#@markdown ## 📍 a. Utils

_LOCAL_FOLDERS = ["training", "validation", "testing"]

def create_local_dirs(reset=False):
    """
    This function is useless if you don't run the code on Google Colab, or any other cloud service.
    Basically, data access is way faster if you copy the data to the local disk rather than a distant server.
    Since the data is accessed multiple times during the training, the choice was made to migrate the data to the local disk.
    There is a possibility to reset the data, in case you want to shuffle your data for the next training.

    Args:
        reset (bool): If True, the folders will be reset.
    """
    if not os.path.isdir(working_directory):
        raise ValueError(f"Working directory '{working_directory}' does not exist.")
    leaves = [inputs_name, annotations_name]
    for r in _LOCAL_FOLDERS:
        for l in leaves:
            path = os.path.join(working_directory, r, l)
            if os.path.isdir(path) and reset:
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

def copy_to(src_folder, dst_folder, files, data_match):
    """
    Copies a list of files from a source folder to a destination folder.

    Args:
        src_folder (str): The source folder.
        dst_folder (str): The destination folder.
        files (list): The list of files to copy.
    """
    for step in [inputs_name, annotations_name]:
        for f in files:
            src_path = os.path.join(src_folder, data_match[step][f])
            dst_path = os.path.join(dst_folder, data_match[step][f])
            shutil.copy(src_path, dst_path)

def check_sum(targets):
    """
    Since we move some fractions of data to some other folders, we need to check that the sum of the ratios is equal to 1.
    Otherwise, we would have some data missing or we would try to read data that doesn't exist.
    """
    acc = sum([i[1] for i in targets])
    return abs(acc - 1.0) < 1e-6

def keys_to_files(source_folder, regex):
    all_files = os.listdir(source_folder)
    matches = {}
    for f in all_files:
        groups = regex.match(f)
        if groups is not None:
            matches[groups[1]] = f
    return matches

def migrate_data(targets, source):
    """
    Copies the content of the source folder to the working directory.
    The percentage of the data to move is defined in the targets list.
    Meant to work with pairs of files.

    Args:
        targets (list): List of tuples. The first element is the name of the folder, the second is the ratio of the data to move.
        source (str): The source folder
    """
    if not check_sum(targets):
        raise ValueError("The sum of the ratios must be equal to 1.")
    folders = [
        (inputs_name     , _IMAGES_REGEX), 
        (annotations_name, _ANNOTATIONS_REGEX)
    ]
    all_data = []
    for f in os.listdir(os.path.join(source, inputs_name)):
        groups = _IMAGES_REGEX.match(f)
        if groups is not None:
            all_data.append(groups[1])
    random.shuffle(all_data)

    data_match = {} # Allows to find matching files with extensions included.
    for f, r in folders: 
        data_match[f] = keys_to_files(os.path.join(source, f), r)

    last = 0
    for target, ratio in targets:
        n = int(len(all_data) * ratio)
        copy_to(
            os.path.join(source, ), 
            os.path.join(working_directory, target), 
            all_data[last:last+n], 
            data_match
        )
        copy_to(
            source, 
            os.path.join(working_directory, target), 
            all_data[last:last+n], 
            data_match
        )
        last += n


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              MAIN FUNCTION                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def main():

    # 1. Running the sanity checks:
    data_sanity = sanity_check(data_folder)
    qc_sanity = True
    if qc_folder is not None:
        qc_sanity = sanity_check(qc_folder)
    
    if not data_sanity:
        print(f"ABORT. 😱 Your {'data' if not data_sanity else 'QC data'} is not consistent. Use the content of the sanity check table above to fix all that and try again.")
    else:
        print(f"👍 Your training data looks alright! (Found {len(os.listdir(os.path.join(data_folder, inputs_name)))} items).")

    if qc_folder is None:
        print("🚨 No QC data provided.")
    elif not qc_sanity:
        print("🚨 Your QC data is not consistent. Use the content of the sanity check table above to fix all that and try again.")
    else:
        print(f"👍 Your QC data looks alright! (Found {len(os.listdir(os.path.join(qc_folder, inputs_name)))} items).")

    # 2. Migrate the data to working directory:
    create_local_dirs(reset_local_data)
    migrate_data([
        ("training", 1.0-validation_percentage),
        ("validation", validation_percentage)
        ], data_folder)
    if qc_folder is not None:
        migrate_data([
            ("testing", 1.0)
            ], qc_folder)
    print("-----------")
    print(f"Training set: {len(os.listdir(os.path.join(working_directory, 'training', inputs_name)))} items.")
    print(f"Validation set: {len(os.listdir(os.path.join(working_directory, 'validation', inputs_name)))} items.")
    if qc_folder is not None:
        print(f"Testing set: {len(os.listdir(os.path.join(working_directory, 'testing', inputs_name)))} items.")


if __name__ == "__main__":
    main()


exit(0)


src_training_dir = os.path.join(root_dir, "train")
src_testing_dir  = os.path.join(root_dir, "test")
# Safety check
if (VALIDATION < 0.0) or (VALIDATION > 1.0):
    raise("Validation percentage must be between 0 and 1.")

# Moves p% of the pairs from src_dir to dst_dir
def move_chunk_to(dst_dir, src_dir, p, just_copy=True):
    dst_images_dir = os.path.join(dst_dir, "images")
    dst_annots_dir = os.path.join(dst_dir, "labels")
    src_images_dir = os.path.join(src_dir, "images")
    src_annots_dir = os.path.join(src_dir, "labels")
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        os.mkdir(dst_images_dir)
        os.mkdir(dst_annots_dir)
    items = sorted(os.listdir(src_images_dir))
    n_items = len(items)
    if n_items == 0:
        raise Exception("Nothing found in source folder.")
    if n_items != len(os.listdir(src_annots_dir)):
        raise Exception(f"The number of images is different from the number of annotations. {n_items} vs {len(os.listdir(src_annots_dir))}.")
    ext = items[0].split('.')[-1]
    n_trans = int(p * n_items)
    print(f"{n_trans} of {n_items}")
    indices = np.random.choice(n_items, n_trans, replace=False)
    task = shutil.copy2 if just_copy else shutil.move
    for index in indices:
        p_from = os.path.join(src_images_dir, items[index])
        p_dest = os.path.join(dst_images_dir, items[index])
        task(p_from, p_dest)
        p_from = os.path.join(src_annots_dir, items[index].replace(ext, 'txt'))
        p_dest = os.path.join(dst_annots_dir, items[index].replace(ext, 'txt'))
        task(p_from, p_dest)


def create_yml():
    with open(os.path.join(working_dir, "data.yml"), 'w') as f:
        f.write(f"train: {working_dir}/train/images\n")
        f.write(f"val: {working_dir}/valid/images\n")
        f.write("\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {str(CLASSES)}")


dirs = ['train', 'valid', 'test']

if FLUSH or not os.path.isdir(os.path.join(working_dir, "train")):
    # Removes all the data from previous attempts:
    for d in dirs:
        full_path = os.path.join(working_dir, d)
        if os.path.isdir(full_path):
            print(f"Deleting {full_path}")
            shutil.rmtree(full_path)

    # We transfer the whole testing directory
    print("> Transfering data...")
    move_chunk_to(os.path.join(working_dir, "test"), src_testing_dir, 1.0)
    print("Testing data transfered...")
    move_chunk_to(os.path.join(working_dir, "train"), src_training_dir, 1.0)
    print("Training data transfered...")
    move_chunk_to(os.path.join(working_dir, "valid"), os.path.join(working_dir, "train"), VALIDATION, False)
    print("Validation data transfered...")
    create_yml()
    print("> Data transfer: DONE.")

"""- The **testing** data will be used to test the network's inference after the training phase.
- The **training** data will be used to train the network.
- The **validation** data (a fraction of the training data) won't ever be seen by the network and will be used during the training phase to measure how the network is learning.

The dataset is structured in the following manner:

```
working_dir
├── test
│   ├── images
│   └── labels
├── train
│   ├── images
│   └── labels
└── valid
    ├── images
    └── labels

```

### The Dataset YAML File

The dataset YAML (`data.yaml`) file containing the path to the training and validation images and labels is already provided. This file will also contain the class names from the dataset.

The dataset contains 3 classes: **'Rod', 'Dividing', 'Microcolony'**.

The following block shows the contents of the `data.yaml` file.

```yaml
train: /content/train/images
val: /content/valid/images

nc: 3
names: ['Rod', 'Dividing', 'Microcolony']
```

### Visualize a Few Ground Truth Images

Before moving forward, let's check out few of the ground truth images.

The current annotations in the text files are in normalized `[x_center, y_center, width, height]` format. Let's write a function that will convert it back to `[x_min, y_min, x_max, y_max]` format.
"""

num_classes = len(CLASSES)
peaks = np.linspace(0, 0.9999, num_classes) * 256
peaks = peaks.astype(np.uint8)

gist_rainbow = plt.colormaps['gist_rainbow']
indices = np.linspace(0, 1, 256)
colors = gist_rainbow(indices)
rgb_colors = (colors[:, :3] * 255).astype(int)

colors = np.array([rgb_colors[i] for i in peaks]).astype(float)

# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)
        width = xmax - xmin
        height = ymax - ymin

        class_name = CLASSES[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=colors[CLASSES.index(class_name)],
            thickness=1
        )
    return image

# Function to plot images with the bounding boxes.
def plot(image_paths, label_paths, num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()

    num_images = len(all_training_images)

    plt.figure()
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()

# Visualize a few training images.
plot(
    image_paths=os.path.join(working_dir, "train", "images", '*'),
    label_paths=os.path.join(working_dir, "train", "labels", '*'),
    num_samples=4,
)

"""## Helper Functions for Logging

Here, we write the helper functions that we need for logging of the results in the notebook while training the models.

Let's create our custom result directories so that we can easily keep track of them and carry out inference using the proper model.
"""

def set_res_dir():
    # Directory to store results
    res_dir_count = len(glob.glob('runs/train/*'))
    print(f"Current number of result directories: {res_dir_count}")
    if TRAIN:
        RES_DIR = f"results_{res_dir_count+1}"
        print(RES_DIR)
    else:
        RES_DIR = f"results_{res_dir_count}"
    print(f"RES_DIR produced: {RES_DIR}")
    return RES_DIR


"""## Training using YOLOV5

The next step is to train the neural network model.

### Train a medium-sized (yolov5m) Model

Training all the layers of the medium model.
"""

monitor_tensorboard()

# RES_DIR = set_res_dir()
# if TRAIN:
#     !python train.py --data {working_dir}/data.yml --weights yolov5m.pt \
#     --epochs {EPOCHS} --batch-size 8 --name {RES_DIR} --img 256

"""- Precision (P) == TP/(TP+FP)
- Recall (R) == TP/(TP+FN)
- F1 score == harmonic average of P and R. (we look at the peak)
"""

def migrate_network(as_name=None):
    abs_path = os.path.join(os.getcwd(), "runs", "train", RES_DIR)
    name = as_name if as_name is not None else RES_DIR
    tgt_dir = os.path.join(root_dir, name)
    if os.path.isdir(tgt_dir):
        print("Removed previous attempt.")
        shutil.rmtree(tgt_dir)
    print("Saved weights to: " + tgt_dir)
    shutil.copytree(abs_path, tgt_dir)

migrate_network()

"""## Check Out the Validation Predictions and Inference

In this section, we will check out the predictions of the validation images saved during training. Along with that, we will also check out inference of images and videos.

### Visualization and Inference Utilities

We will visualize the validation prediction images that are saved during training. The following is the function for that.
"""

# Function to show validation predictions saved during training.
def show_valid_results(RES_DIR):
    # !ls runs/train/{RES_DIR}
    EXP_PATH = f"runs/train/{RES_DIR}"
    validation_pred_images = glob.glob(f"{EXP_PATH}/*_pred.jpg")
    print(validation_pred_images)
    for pred_image in validation_pred_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.show()

"""The following functions are for carrying out inference on images and videos."""

# Helper function for inference on images.
def inference(RES_DIR, data_path):
    # Directory to store inference results.
    infer_dir_count = len(glob.glob('runs/detect/*'))
    print(f"Current number of inference detection directories: {infer_dir_count}")
    INFER_DIR = f"inference_{infer_dir_count+1}"
    print(INFER_DIR)
    # Inference on images.
    # !python detect.py --weights runs/train/{RES_DIR}/weights/best.pt --img 256\
    # --source {data_path} --name {INFER_DIR} --line-thickness 1 --hide-labels --hide-conf
    return INFER_DIR

"""We may also need to visualize images in any of the directories. The following function accepts a directory path and plots all the images in them."""

def visualize(INFER_DIR, n_items=-1):
# Visualize inference images.
    INFER_PATH = f"runs/detect/{INFER_DIR}"
    infer_images = glob.glob(f"{INFER_PATH}/*.png")
    if n_items > 0:
        random.shuffle(infer_images)
        infer_images = infer_images[:n_items]
    print(infer_images)
    for pred_image in infer_images:
        image = cv2.imread(pred_image)
        plt.figure(figsize=(19, 16))
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
        plt.show()

"""**Visualize validation prediction images.**"""

show_valid_results(RES_DIR)

"""### Inference
In this section, we will carry out inference on unseen images from the `test` directory.

### Run on testing set (Inference on Images)

**To carry out inference on images, we just need to provide the directory path where all the images are stored, and inference will happen on all images automatically.**
"""

# Inference on images.
IMAGE_INFER_DIR = inference(RES_DIR, os.path.join(working_dir, "test", "images"))
print(f"Predictions exported to: {IMAGE_INFER_DIR}")

visualize(IMAGE_INFER_DIR, 5)

"""## Get the data back to Google Drive"""

def migrate_results():
    cwd = os.getcwd()
    abs_path = os.path.join(cwd, "runs", "detect", IMAGE_INFER_DIR)
    print(f"Using results from: {abs_path}")
    tgt_path = os.path.join(src_testing_dir, "predictions")
    if os.path.isdir(tgt_path):
        print("Removing previous attempt")
        shutil.rmtree(tgt_path)
    shutil.copytree(abs_path, tgt_path)
    print("Moved results to `tests/predictions`")

migrate_results()