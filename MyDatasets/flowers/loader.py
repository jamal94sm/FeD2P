
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
import tarfile
import urllib.request
from PIL import Image
from datasets import Dataset, DatasetDict
import random

def download_and_extract_flowers17(data_dir="17flowers"):
    url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
    tgz_path = os.path.join(data_dir, "17flowers.tgz")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(tgz_path):
        print("Downloading Oxford Flowers 17 dataset...")
        urllib.request.urlretrieve(url, tgz_path)
        print("Download complete.")

    with tarfile.open(tgz_path) as tar:
        tar.extractall(path=data_dir)
        print("Extraction complete.")

def load_dataset(num_train_samples, num_test_samples, num_public_samples, data_dir="17flowers/jpg"):
    # Download and extract if needed
    download_and_extract_flowers17(os.path.dirname(data_dir))

    # Load image paths
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
    image_paths = [os.path.join(data_dir, f) for f in image_files]

    # Assign labels: 80 images per class, 17 classes
    labels = [i // 80 for i in range(len(image_paths))]

    # Shuffle and split
    combined = list(zip(image_paths, labels))
    random.seed(42)
    random.shuffle(combined)

    images, labels = zip(*combined)

    train_data = {"image": images[:num_train_samples], "label": labels[:num_train_samples]}
    test_data = {"image": images[num_train_samples:num_train_samples + num_test_samples],
                 "label": labels[num_train_samples:num_train_samples + num_test_samples]}
    public_data = {"image": images[num_train_samples + num_test_samples:num_train_samples + num_test_samples + num_public_samples],
                   "label": labels[num_train_samples + num_test_samples:num_train_samples + num_test_samples + num_public_samples]}

    # Convert to Hugging Face Dataset and preprocess
    train_dataset = prepare_dataset(Dataset.from_dict(train_data))
    test_dataset = prepare_dataset(Dataset.from_dict(test_data))
    public_dataset = prepare_dataset(Dataset.from_dict(public_data))

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    public_data = DatasetDict({'train': public_dataset, 'test': None})

    num_classes = 17
    name_classes = [f"class_{i}" for i in range(num_classes)]


######################################################################################################
######################################################################################################
