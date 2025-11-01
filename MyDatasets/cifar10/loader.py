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

def build_public_data(full_train_data, num_classes, num_samples):
    # Ensure correct column names
    if "image" not in full_train_data.column_names:
        full_train_data = full_train_data.rename_column(full_train_data.column_names[0], "image")
    if "label" not in full_train_data.column_names:
        full_train_data = full_train_data.rename_column(full_train_data.column_names[1], "label")

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(full_train_data["label"]):
        class_to_indices[label].append(idx)

    public_indices = []
    samples_per_class = int(num_samples // num_classes)
    for label in range(num_classes):
        selected = random.sample(class_to_indices[label], min(samples_per_class, len(class_to_indices[label])))
        public_indices.extend(selected)

    # Select samples directly from the dataset
    public_train = full_train_data.select(public_indices)
    public_train.set_format("torch", columns=["image", "label"])  # Convert to torch tensors

    # Normalize image tensors to float32
    def normalize(example):
        example["image"] = example["image"].float() / 255.0
        return example

    public_train = public_train.map(normalize)

    public_test = None

    return DatasetDict({'train': public_train, 'test': public_test})

######################################################################################################
######################################################################################################

def load_dataset(num_train_samples, num_test_samples):
    loaded_dataset = hf_load_dataset("cifar10", split=['train[:50%]', 'test[:50%]'])

    name_classes = loaded_dataset[0].features["label"].names
    num_classes = len(name_classes)

    public_data = build_public_data(loaded_dataset[0], num_classes, num_train_samples)

    train_data = ddf(loaded_dataset[0][shuffling(loaded_dataset[0].num_rows, num_train_samples)])
    test_data = ddf(loaded_dataset[1][shuffling(loaded_dataset[1].num_rows, num_test_samples)])

    dataset = DatasetDict({"train": train_data, "test": test_data})

    if "image" not in dataset["train"].column_names:
        dataset = dataset.rename_column(dataset["train"].column_names[0], 'image')
    if "label" not in dataset["train"].column_names:
        dataset = dataset.rename_column(dataset["train"].column_names[1], 'label')

    dataset.set_format("torch", columns=["image", "label"])

    if dataset["train"]["image"].max() > 1:
        dataset = dataset.map(normalization, batched=True)


    return dataset, num_classes, name_classes, public_data

######################################################################################################
######################################################################################################











