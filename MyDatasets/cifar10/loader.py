import numpy as np
import datasets
import torch
from datasets import load_dataset as hf_load_dataset, DatasetDict, Dataset
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
def prepare_dataset(data):
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    data.set_format("torch", columns=["image", "label"])

    # Normalize image tensors to float32 in [0, 1]
    def normalize(example):
        example["image"] = example["image"].float() / 255.0
        return example

    data = data.map(normalize)

    return data



################################################################

def load_dataset(num_train_samples, num_test_samples, num_public_samples):
    try:
        # Load full training set
        full_train = hf_load_dataset("cifar10", split="train[:100%]", download_mode="reuse_dataset_if_exists")
    except Exception as e:
        print("Local cache not found or failed to load. Trying to download from internet...")
        full_train = hf_load_dataset("cifar10", split="train[:100%]")

    name_classes = full_train.features["label"].names
    num_classes = len(name_classes)

    # Shuffle full training set once
    full_train = full_train.shuffle(seed=42)

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




