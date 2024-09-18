import os
import cv2
import random
import numpy as np
import shutil
import tifffile
import re
import math

from yolov5 import train

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""

# YOLOv5 (Object detection with PyTorch)
----------------------------------------

Before starting using this script, please make sure that:
    - You have some annotated images.
    - You have the required modules installed.
    - You cloned/downloaded the YOLOv5 repository (https://github.com/ultralytics/yolov5.git).
    - You created an empty file named "__init__.py" in the yolov5 folder.

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

data_folder       = "/home/benedetti/Desktop/eaudissect/deep-bacs-tiff/train/"
qc_folder         = "/home/benedetti/Desktop/eaudissect/deep-bacs-tiff/qc_data/"
inputs_name       = "images"
annotations_name  = "labels"
models_path       = "/home/benedetti/Desktop/eaudissect/yolo_working/models"
working_directory = "/home/benedetti/Desktop/eaudissect/yolo_local"
model_name_prefix = "YOLOv5"
reset_local_data  = True
# preview_data      = True

#@markdown ## 📍 b. Network architecture

validation_percentage = 0.15
batch_size            = 16
epochs                = 50
classes_names         = ["Rod", "Dividing", "Microcolony"]
optimizer             = 'Adam'
learning_rate         = 0.0001

#@markdown ## 📍 c. Constants

_IMAGES_REGEX = re.compile(r"(.+)\.(tif|tiff)$")
_ANNOTATIONS_REGEX = re.compile(r"(.+)\.(txt)$")
_N_CLASSES = len(classes_names)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            SANITY CHECK                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 2. SANITY CHECK

"""
List of all the points checked during the sanity check:
    - [X] Is the content of folders valid (only files with the correct extension)?
    - [X] Does every image have its corresponding annotation?
    - [X] Is each annotation correctly formatted (class, x, y, width, height)?
    - [X] Does every annotation have at least one bounding box?
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

def t_xor(t1, t2):
    return tuple([i1 if i2 is None else i2 for (i1, i2) in zip(t1, t2)])

def make_tuple(arg, pos, size):
    return tuple([arg if i == pos else None for i in range(size)])

def files_as_keys(root_folder, sources):
    """
    To handle the fact that we have different extensions for the same pair, we use this function producing a dictionary.
    Keys are the files without their extensions, values are tuples containing the files with their extensions.
    Example: 'file': ('file.png', 'file.txt').

    Args:
        root_folder (str): The root folder, containing the `inputs_name` and `annotations_name` folders.
        sources (folder, regex): Tuples containing sub-folder name and its associated regex pattern.
    """
    if len(sources) == 0:
        raise ValueError("No sources provided.")
    # Removing extensions to build keys.
    matches = {}
    for i, (subfolder, regex) in enumerate(sources):
        for f in os.listdir(os.path.join(root_folder, subfolder)):
            groups = regex.match(f)
            if groups is  None:
                continue
            handles = make_tuple(f, i, len(sources))
            key = groups[1]
            if key not in matches:
                matches[key] = handles
            else:
                matches[key] = t_xor(matches[key], handles)
    return matches

def check_files_keys(matches):
    """
    Checks if the keys of the dictionary produced by `files_as_keys` are unique.

    Args:
        matches (dict): The dictionary produced by `files_as_keys`.
    """
    errors = set()
    for key, values in matches.items():
        if None in values:
            errors.add(key)
    if len(errors) > 0:
        print(f"Errors found: {errors}")
    return len(errors) == 0

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

def copy_to(src_folder, folders_name, files_name, to_copy, dst_folder, usage):
    """
    Copies a list of files from a source folder to a destination folder.

    Args:
        src_folder (str): The source folder.
        dst_folder (str): The destination folder.
        files (list): The list of files to copy.
    """
    for key in to_copy:
        for i, f in enumerate(folders_name):
            src_path = os.path.join(src_folder, f, files_name[key][i])
            dst_path = os.path.join(dst_folder, usage, f, files_name[key][i])
            shutil.copy2(src_path, dst_path)

def check_sum(targets):
    """
    Since we move some fractions of data to some other folders, we need to check that the sum of the ratios is equal to 1.
    Otherwise, we would have some data missing or we would try to read data that doesn't exist.
    """
    acc = sum([i[1] for i in targets])
    return abs(acc - 1.0) < 1e-6

"""
Local structure of the data:
----------------------------

working_directory
├── test
│   ├── images
│   └── labels
├── train
│   ├── images
│   └── labels
└── valid
    ├── images
    └── labels
"""

def migrate_data(targets, source, tuples):
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
    folders = [inputs_name, annotations_name]
    all_data = list(tuples.keys())
    random.shuffle(all_data) # Avoid taking twice the same data by shuffling.

    last = 0
    for target, ratio in targets:
        n = int(len(all_data) * ratio)
        copy_to(
            source, # folder with 'images' and 'labels'
            folders, # ['images', 'labels']
            tuples, # files in every folder
            all_data[last:last+n], # keys to copy
            working_directory, # destination root
            target # destination sub-folder (training, validation, testing)
        )
        last += n


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            PREPARE TRAINING                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#@markdown # ⭐ 4. PREPARE TRAINING

#@markdown ## 📍 a. Utils

def create_dataset_yml():
    with open(os.path.join(working_directory, "data.yml"), 'w') as f:
        f.write(f"train: {os.path.join(working_directory, 'training', inputs_name)}\n")
        f.write(f"val: {os.path.join(working_directory, 'validation', inputs_name)}\n")
        f.write("\n")
        f.write(f"nc: {_N_CLASSES}\n")
        f.write(f"names: {str(classes_names)}\n")

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


#@markdown ## 📍 b. Data visualization

def get_classes_color():
    """
    Produces a list of RGB colors (0-255, not 0-1) for each class.
    Colors are based on the 'gist_rainbow' colormap.
    """
    peaks = np.linspace(0, 0.9999, _N_CLASSES) * 256
    peaks = peaks.astype(np.uint8)
    gist_rainbow = plt.colormaps['gist_rainbow']
    indices = np.linspace(0, 1, 256)
    colors = gist_rainbow(indices)
    rgb_colors = (colors[:, :3] * 255).astype(int)
    return np.array([rgb_colors[i] for i in peaks]).astype(float)

def yolo2bbox(bboxes):
    """
    Converts bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
    This is only useful for visualization purposes.
    """
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels, colors):
    # Need the image height and width to denormalize the bounding box coordinates.
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)
        class_name = classes_names[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=colors[classes_names.index(class_name)],
            thickness=1
        )
    return image

def plot(source_folder, subfolders, files_name, colors, n_items=5):
    all_files = list(files_name.keys())
    random.shuffle(all_files)
    count, i = 0, 0
    plt.figure()

    while (i < len(all_files)) and (count <= n_items):
        image_path = os.path.join(source_folder, subfolders[0], files_name[all_files[i]][0])
        label_path = os.path.join(source_folder, subfolders[1], files_name[all_files[i]][1])
        # We want to plot items from the training set.
        # -> We must check that the file has not been moved to the validation set.
        if (not os.path.isfile(image_path)) or (not os.path.isfile(label_path)):
            i += 1
            continue
        image = tifffile.imread(image_path)
        with open(label_path, 'r') as f:
            bboxes = [] # List of bounding boxes.
            labels = [] # List of labels corresponding to the bounding boxes.
            label_lines = f.readlines()
            for label_line in label_lines:
                if len(label_line.strip()) == 0 or label_line.startswith('#'):
                    continue
                items = label_line.strip().split(' ')
                label = int(items[0])
                x_c   = float(items[1])
                y_c   = float(items[2])
                w     = float(items[3])
                h     = float(items[4])
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels, colors)
        plt.subplot(int(math.sqrt(n_items))+1, int(math.sqrt(n_items))+1, count+1)
        plt.imshow(result_image[:, :])
        plt.axis('off')
        i += 1
        count += 1

    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              MAIN FUNCTION                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def tests():
    from pprint import pprint
    files_tuples = files_as_keys(data_folder, [
        (inputs_name, _IMAGES_REGEX),
        (annotations_name, _ANNOTATIONS_REGEX)
    ])
    colors = get_classes_color()
    plot(
        os.path.join(working_directory, "training"), 
        [inputs_name, annotations_name], 
        files_tuples,
        colors
    )


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
    files_tuples = files_as_keys(data_folder, [
        (inputs_name, _IMAGES_REGEX),
        (annotations_name, _ANNOTATIONS_REGEX)
    ])
    if not check_files_keys(files_tuples):
        return False
    create_local_dirs(reset_local_data)
    migrate_data([
        ("training", 1.0-validation_percentage),
        ("validation", validation_percentage)
        ], data_folder, files_tuples)
    if qc_folder is not None:
        qc_tuples = files_as_keys(qc_folder, [
            (inputs_name, _IMAGES_REGEX),
            (annotations_name, _ANNOTATIONS_REGEX)
        ])
        if not check_files_keys(qc_tuples):
            return False
        migrate_data([
            ("testing", 1.0)
            ], qc_folder, qc_tuples)
    print("-----------")
    print(f"Training set: {len(os.listdir(os.path.join(working_directory, 'training', inputs_name)))} items.")
    print(f"Validation set: {len(os.listdir(os.path.join(working_directory, 'validation', inputs_name)))} items.")
    if qc_folder is not None:
        print(f"Testing set: {len(os.listdir(os.path.join(working_directory, 'testing', inputs_name)))} items.")
    
    # 3. Prepare for training:
    create_dataset_yml()
    colors = get_classes_color()
    plot(
        os.path.join(working_directory, "training"), 
        [inputs_name, annotations_name], 
        files_tuples,
        colors
    )
    v = get_version()
    version_name = f"{model_name_prefix}-V{str(v).zfill(3)}"
    
    # 4. Launch the training:
    train.run(
        data=os.path.join(working_directory, "data.yml"),
        epochs=epochs,
        batch_size=batch_size,
        project=models_path,
        name=version_name
    )


if __name__ == "__main__":
    # tests()
    main()


exit(0)



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