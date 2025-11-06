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
    transforms.Resize((32, 32)),
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

def load_dataset(num_train_samples, num_test_samples, num_public_samples):
    full_dataset = hf_load_dataset("mikewang/EuroSAT")["train"]
    full_dataset = full_dataset.rename_column("image_path", "image")
    full_dataset = full_dataset.rename_column("class", "label")

    unique_classes = sorted(set(full_dataset["label"]))
    class_label = ClassLabel(names=unique_classes)
    num_classes = len(unique_classes)

    full_dataset = full_dataset.cast_column("image", datasets.Image())

    # Shuffle full dataset
    full_dataset = full_dataset.shuffle(seed=42)

    # Select slices
    train_slice = full_dataset.select(range(0, num_train_samples))
    test_slice = full_dataset.select(range(num_train_samples, num_train_samples + num_test_samples))
    public_slice = full_dataset.select(range(num_train_samples + num_test_samples,
                                             num_train_samples + num_test_samples + num_public_samples))

    # Prepare datasets
    train_data = prepare_dataset(train_slice, class_label)
    test_data = prepare_dataset(test_slice, class_label)
    public_train_data = prepare_dataset(public_slice, class_label)

    dataset = DatasetDict({
        "train": train_data,
        "test": test_data
    })

    public_data = DatasetDict({
        "train": public_train_data,
        "test": None
    })

    return dataset, num_classes, unique_classes, public_data
