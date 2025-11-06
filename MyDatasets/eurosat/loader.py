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
    '''
    try:
        # Try loading from local cache
        full_dataset = hf_load_dataset(
            "mikewang/EuroSAT",
            split="train[:100%]",
            download_mode="reuse_dataset_if_exists"
        )
    except Exception as e:
        print("Local cache not found or failed to load. Downloading from internet...")
        full_dataset = hf_load_dataset("mikewang/EuroSAT", split="train[:100%]")

    '''
    full_dataset = hf_load_dataset("mikewang/EuroSAT", split="train[:100%]")
    
    # Rename columns for consistency
    full_dataset = full_dataset.rename_column("image_path", "image")
    full_dataset = full_dataset.rename_column("class", "label")

    # Get class info
    unique_classes = sorted(set(full_dataset["label"]))
    class_label = ClassLabel(names=unique_classes)
    num_classes = len(unique_classes)

    # Cast image column to datasets.Image
    full_dataset = full_dataset.cast_column("image", datasets.Image())

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, example in enumerate(full_dataset):
        class_to_indices[example["label"]].append(idx)

    # Make a copy of indices to avoid overlap
    available_indices_by_class = {label: indices.copy() for label, indices in class_to_indices.items()}

    def sample_uniformly(total_samples, available_indices_by_class):
        samples_per_class = total_samples // num_classes
        selected_indices = []
        for label in unique_classes:
            indices = available_indices_by_class[label]
            if len(indices) < samples_per_class:
                raise ValueError(f"Not enough samples in class '{label}' to fulfill request.")
            selected = random.sample(indices, samples_per_class)
            selected_indices.extend(selected)
            # Remove selected indices to avoid reuse
            available_indices_by_class[label] = list(set(indices) - set(selected))
        return selected_indices

    # Sample non-overlapping sets
    train_indices = sample_uniformly(num_train_samples, available_indices_by_class)
    test_indices = sample_uniformly(num_test_samples, available_indices_by_class)
    public_indices = sample_uniformly(num_public_samples, available_indices_by_class)

    # Select and prepare datasets
    train_slice = full_dataset.select(train_indices)
    test_slice = full_dataset.select(test_indices)
    public_slice = full_dataset.select(public_indices)

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
