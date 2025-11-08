import numpy as np
import datasets
import torch
from datasets import load_dataset as hf_load_dataset, DatasetDict
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
    transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to RGB
])

def resize_and_repeat(batch):
    batch["image"] = [resize_transform(img) for img in batch["image"]]
    return batch

def normalization(batch):
    batch["image"] = [transforms.ToTensor()(img) for img in batch["image"]]
    return batch





######################################################################################################
######################################################################################################



def prepare_dataset(data):
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    data = data.cast_column("image", datasets.Image())

    # Resize before converting to torch
    data = data.map(resize_and_repeat, batched=True)

    # Convert to tensor
    data = data.map(normalization, batched=True)

    data.set_format("torch", columns=["image", "label"])

    return data



######################################################################################################
######################################################################################################
import os

def load_dataset(num_train_samples, num_test_samples, num_public_samples):
    try:
        # Load full training and test sets
        loaded_dataset = hf_load_dataset(
            "fashion_mnist",
            split=["train[:100%]", "test[:100%]"],
            download_mode="reuse_dataset_if_exists"
        )
    except Exception as e:
        print("Local cache not found or failed to load. Trying to download from internet...")
        loaded_dataset = hf_load_dataset(
            "fashion_mnist",
            split=["train[:100%]", "test[:100%]"]
        )

    name_classes = loaded_dataset[0].features["label"].names
    name_classes = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]
    num_classes = len(name_classes)

    # Shuffle full training set once
    full_train = loaded_dataset[0].shuffle(seed=42)

    # Slice non-overlapping subsets
    train_slice = full_train.select(range(0, num_train_samples))
    test_slice = full_train.select(range(num_train_samples, num_train_samples + num_test_samples))
    public_slice = full_train.select(range(num_train_samples + num_test_samples,
                                           num_train_samples + num_test_samples + num_public_samples))

    # Prepare each subset
    train_data = prepare_dataset(train_slice)
    test_data = prepare_dataset(test_slice)
    public_train_data = prepare_dataset(public_slice)

    dataset = DatasetDict({"train": train_data, "test": test_data})
    public_data = DatasetDict({'train': public_train_data, 'test': None})


    return dataset, num_classes, name_classes, public_data
######################################################################################################
######################################################################################################









