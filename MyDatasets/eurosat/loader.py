import numpy as np
import datasets
import torch
from datasets import ClassLabel
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import random



######################################################################################################
######################################################################################################
def ddf(x):
    x = datasets.Dataset.from_dict(x)
    x.set_format("torch")
    return x

######################################################################################################
######################################################################################################
def shuffling(a, b):
    return np.random.randint(0, a, b)

######################################################################################################
######################################################################################################
def normalization(batch):
    normal_image = batch["image"] / 255
    return {"image": normal_image, "label": batch["label"]}

######################################################################################################
######################################################################################################

def build_public_data(full_dataset, class_label, num_classes, num_samples, transform):
    samples_per_class = int(num_samples // num_classes)

    # Group images by class
    class_to_images = defaultdict(list)
    for example in full_dataset:
        label = class_label.str2int(example["label"])
        class_to_images[label].append(example["image"])

    public_images = []
    public_labels = []

    for label in range(num_classes):
        selected = random.sample(class_to_images[label], min(samples_per_class, len(class_to_images[label])))
        for image_path in selected:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)
            public_images.append(image_tensor)
            public_labels.append(label)

    public_train = datasets.Dataset.from_dict({'image': public_images, 'label': public_labels})
    public_test = None

    return datasets.DatasetDict({'train': ddf(public_train.to_dict()), 'test': public_test})


######################################################################################################
######################################################################################################

def load_dataset(num_train_samples, num_test_samples):
    full_dataset = datasets.load_dataset("mikewang/EuroSAT")["train"]
    full_dataset = full_dataset.rename_column("image_path", "image")
    full_dataset = full_dataset.rename_column("class", "label")

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create label encoder from full dataset
    unique_classes = sorted(set(full_dataset["label"]))
    class_label = ClassLabel(names=unique_classes)
    num_classes = len(unique_classes)

    # Build public data
    public_data = build_public_data(full_dataset, class_label, num_classes, num_train_samples, transform)

    # Shuffle and select indices
    train_indices = shuffling(full_dataset.num_rows, num_train_samples)
    test_indices = shuffling(full_dataset.num_rows, num_test_samples)

    train_dataset = full_dataset.select(train_indices)
    test_dataset = full_dataset.select(test_indices)

    def map_label(example):
        example["label"] = class_label.str2int(example["label"])
        return example

    def load_image(example):
        image_path = example["image"]
        image = Image.open(image_path).convert("RGB")
        example["image"] = transform(image)
        return example

    train_dataset = train_dataset.map(map_label).map(load_image)
    test_dataset = test_dataset.map(map_label).map(load_image)

    dataset = datasets.DatasetDict({
        "train": ddf(train_dataset.to_dict()),
        "test": ddf(test_dataset.to_dict())
    })

    dataset.set_format("torch", columns=["image", "label"])

    return dataset, num_classes, unique_classes, public_data

######################################################################################################
######################################################################################################



