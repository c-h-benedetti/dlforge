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