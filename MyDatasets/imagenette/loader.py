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
def normalization(batch):
    return {
        "image": [img / 255.0 for img in batch["image"]],
        "label": batch["label"]
    }

######################################################################################################
######################################################################################################

def build_public_data(full_dataset, num_classes, num_samples):
    samples_per_class = int(num_samples // num_classes)

    # Group images by class
    class_to_images = defaultdict(list)
    for example in full_dataset:
        label = example["label"]
        class_to_images[label].append(example["image"])

    public_images = []
    public_labels = []

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # SVHN images are already 32x32
        transforms.ToTensor()
    ])

    for label in range(num_classes):
        selected = random.sample(class_to_images[label], min(samples_per_class, len(class_to_images[label])))
        for img in selected:
            image_tensor = transform(img) / 255.0
            public_images.append(image_tensor)
            public_labels.append(label)

    public_train = datasets.Dataset.from_dict({'image': public_images, 'label': public_labels})
    public_test = None

    return datasets.DatasetDict({'train': ddf(public_train.to_dict()), 'test': public_test})

######################################################################################################
######################################################################################################

from datasets import load_dataset, DatasetDict
import random

def load_dataset(num_train_samples, num_test_samples, num_public_samples):
    try:
        # Try loading from cache
        dataset = load_dataset("randall-lab/imagenette", split="train", trust_remote_code=True)
    except Exception as e:
        print("Failed to load Imagenette from cache. Trying to download...")
        dataset = load_dataset("randall-lab/imagenette", split="train", trust_remote_code=True)

    # Shuffle and split
    dataset = dataset.shuffle(seed=42)

    train_slice = dataset.select(range(0, num_train_samples))
    test_slice = dataset.select(range(num_train_samples, num_train_samples + num_test_samples))
    public_slice = dataset.select(range(num_train_samples + num_test_samples,
                                        num_train_samples + num_test_samples + num_public_samples))

    # Apply preprocessing
    train_data = prepare_dataset(train_slice)
    test_data = prepare_dataset(test_slice)
    public_train_data = prepare_dataset(public_slice)

    dataset_dict = DatasetDict({"train": train_data, "test": test_data})
    public_data = DatasetDict({'train': public_train_data, 'test': None})

    num_classes = 10
    name_classes = [
        "tench", "English springer", "cassette player", "chain saw", "church",
        "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
    ]

    print(f"Returning Imagenette dataset with {len(train_data)} training samples, {len(test_data)} test samples.")

    return dataset_dict, num_classes, name_classes, public_data

    return dataset, num_classes, name_classes, public_data

######################################################################################################
######################################################################################################
