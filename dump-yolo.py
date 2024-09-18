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


"""

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


def plot(source_folder, subfolders, files_name, n_items=5):
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



exit(0)




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