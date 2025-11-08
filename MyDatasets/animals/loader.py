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

def load_dataset(num_train_samples, num_test_samples):

    # Load SVHN dataset with config name
    loaded_dataset = datasets.load_dataset("svhn", "cropped_digits", split=["train", "test"])

    # Shuffle and select samples
    train_indices = shuffling(loaded_dataset[0].num_rows, num_train_samples)
    test_indices = shuffling(loaded_dataset[1].num_rows, num_test_samples)

    # Select subsets
    train_dataset = loaded_dataset[0].select(train_indices)
    test_dataset = loaded_dataset[1].select(test_indices)

    # Decode image column to actual image objects
    train_dataset = train_dataset.cast_column("image", datasets.Image())
    test_dataset = test_dataset.cast_column("image", datasets.Image())

    # Convert to DatasetDict
    dataset = datasets.DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Set format for PyTorch
    dataset.set_format("torch", columns=["image", "label"])


    dataset = dataset.map(normalization, batched=True)

    # Get class names
    name_classes = loaded_dataset[0].features["label"].names
    num_classes = len(name_classes)

    # Build public data
    public_data = build_public_data(loaded_dataset[0], num_classes, num_train_samples)

    return dataset, num_classes, name_classes, public_data

######################################################################################################
######################################################################################################