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



def prepare_dataset(data, num_classes, samples_per_class):
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    data = data.cast_column("image", datasets.Image())

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(data["label"]):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label in range(num_classes):
        indices = class_to_indices[label]
        selected = random.sample(indices, min(samples_per_class, len(indices)))
        selected_indices.extend(selected)

    sampled_data = data.select(selected_indices)

    # Resize before converting to torch
    sampled_data = sampled_data.map(resize_and_repeat, batched=True)

    # Convert to tensor
    sampled_data = sampled_data.map(normalization, batched=True)

    sampled_data.set_format("torch", columns=["image", "label"])


    return sampled_data

######################################################################################################
######################################################################################################

def load_dataset(num_train_samples, num_test_samples, num_public_samples):
    loaded_dataset = hf_load_dataset("fashion_mnist", split=["train[:100%]", "test[:100%]"])

    name_classes = loaded_dataset[0].features["label"].names
    num_classes = len(name_classes)

    samples_per_class_train = num_train_samples // num_classes
    samples_per_class_test = num_test_samples // num_classes
    samples_per_class_public = num_public_samples // num_classes

    train_data = prepare_dataset(loaded_dataset[0], num_classes, samples_per_class_train)
    test_data = prepare_dataset(loaded_dataset[1], num_classes, samples_per_class_test)
    public_train_data = prepare_dataset(loaded_dataset[0], num_classes, samples_per_class_public)

    dataset = DatasetDict({"train": train_data, "test": test_data})
    public_data = DatasetDict({'train': public_train_data, 'test': None})

    return dataset, num_classes, name_classes, public_data

######################################################################################################
######################################################################################################









