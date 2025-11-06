import numpy as np
import datasets
import torch
from datasets import load_dataset as hf_load_dataset, DatasetDict, ClassLabel
from collections import defaultdict
import random
import torchvision.transforms as transforms

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

resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
])

def resize_and_repeat(batch):
    batch["image"] = [resize_transform(img) for img in batch["image"]]
    return batch

def normalization(batch):
    batch["image"] = [transforms.ToTensor()(img) for img in batch["image"]]
    return batch

######################################################################################################
######################################################################################################

def prepare_dataset(data, class_label=None):
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    data = data.cast_column("image", datasets.Image())

    if class_label:
        def encode_label(example):
            example["label"] = class_label.str2int(example["label"])
            return example
        data = data.map(encode_label)

    data = data.map(resize_and_repeat, batched=True)
    data = data.map(normalization, batched=True)
    data.set_format("torch", columns=["image", "label"])

    return data

######################################################################################################
######################################################################################################

def load_dataset(num_train_samples, num_test_samples):
    full_dataset = hf_load_dataset("mikewang/EuroSAT")["train"]
    full_dataset = full_dataset.rename_column("image_path", "image")
    full_dataset = full_dataset.rename_column("class", "label")

    unique_classes = sorted(set(full_dataset["label"]))
    class_label = ClassLabel(names=unique_classes)
    num_classes = len(unique_classes)

    full_dataset = full_dataset.cast_column("image", datasets.Image())

    # Shuffle and select indices
    train_indices = shuffling(full_dataset.num_rows, num_train_samples)
    test_indices = shuffling(full_dataset.num_rows, num_test_samples)

    train_dataset = full_dataset.select(train_indices)
    test_dataset = full_dataset.select(test_indices)

    train_data = prepare_dataset(train_dataset, class_label)
    test_data = prepare_dataset(test_dataset, class_label)

    dataset = DatasetDict({
        "train": train_data,
        "test": test_data
    })

    # Build public data
    samples_per_class = int(num_train_samples // num_classes)
    class_to_images = defaultdict(list)
    for example in full_dataset:
        label = class_label.str2int(example["label"])
        class_to_images[label].append(example["image"])

    public_images = []
    public_labels = []

    for label in range(num_classes):
        selected = random.sample(class_to_images[label], min(samples_per_class, len(class_to_images[label])))
        public_images.extend(selected)
        public_labels.extend([label] * len(selected))

    public_raw = datasets.Dataset.from_dict({'image': public_images, 'label': public_labels})
    public_raw = public_raw.cast_column("image", datasets.Image())
    public_train_data = prepare_dataset(public_raw)

    public_data = DatasetDict({'train': public_train_data, 'test': None})

    return dataset, num_classes, unique_classes, public_data
