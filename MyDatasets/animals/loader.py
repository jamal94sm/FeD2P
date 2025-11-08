
import numpy as np
import datasets
import torch
import random
from collections import defaultdict
import random
from PIL import Image
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

from datasets import load_dataset as hf_load_dataset, DatasetDict
import random

def load_dataset(num_train_samples, num_test_samples, num_public_samples):
    try:
        dataset_dict = hf_load_dataset(
            "Rapidata/Animals-10",
            cache_dir="/home/shahab33/scratch/huggingface_cache",
            local_files_only=True  # offline-first
        )
    except Exception as e:
        print("Failed to load Animals-10 dataset:", e)
        raise RuntimeError("Animals-10 dataset could not be loaded.")

    if dataset_dict is None or "train" not in dataset_dict:
        raise RuntimeError("Animals-10 dataset missing or invalid.")

    # Shuffle and slice
    full_data = dataset_dict["train"].shuffle(seed=42)
    train_slice = full_data.select(range(0, num_train_samples))
    test_slice = full_data.select(range(num_train_samples, num_train_samples + num_test_samples))
    public_slice = full_data.select(range(num_train_samples + num_test_samples,
                                          num_train_samples + num_test_samples + num_public_samples))

    train_data = prepare_dataset(train_slice)
    test_data = prepare_dataset(test_slice)
    public_train_data = prepare_dataset(public_slice)

    dataset = DatasetDict({"train": train_data, "test": test_data})
    public_data = DatasetDict({'train': public_train_data, 'test': None})

    num_classes = 10
    name_classes = [
        "butterfly", "cat", "chicken", "cow", "dog",
        "elephant", "horse", "sheep", "spider", "squirrel"
    ]

    return dataset, num_classes, name_classes, public_data
######################################################################################################
######################################################################################################
