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
    # Ensure canonical column names
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    # Cast image to PIL-backed datasets.Image
    data = data.cast_column("image", datasets.Image())

    # Encode labels only if they're strings (if already ClassLabel -> keep as is)
    if class_label is not None:
        label_feature = data.features.get("label", None)
        needs_encoding = True
        if isinstance(label_feature, ClassLabel):
            needs_encoding = False
        else:
            # If not a ClassLabel, check first element's type (string vs int)
            sample_val = data[0]["label"]
            if isinstance(sample_val, int):
                needs_encoding = False

        if needs_encoding:
            def encode_label(example):
                example["label"] = class_label.str2int(example["label"])
                return example
            data = data.map(encode_label)

    # Preprocess images
    data = data.map(resize_and_repeat, batched=True)
    data = data.map(normalization, batched=True)
    data.set_format("torch", columns=["image", "label"])
    return data

######################################################################################################
######################################################################################################

def _compute_counts_per_class(total_samples, num_classes, class_order, seed=None):
    """
    Distribute total_samples as uniformly as possible across classes.
    Any remainder is distributed by adding 1 to the first 'remainder' classes
    after shuffling the class order for fairness.
    """
    rng = random.Random(seed)
    base = total_samples // num_classes
    rem = total_samples % num_classes
    order = list(class_order)
    rng.shuffle(order)

    counts = {c: base for c in class_order}
    for c in order[:rem]:
        counts[c] += 1
    return counts

def _sample_indices_by_counts(available_indices_by_class, counts_per_class, seed=None):
    """
    Sample the requested number of indices per class from the available pool.
    Removes the sampled indices from the available pool to prevent overlap
    across splits.
    """
    rng = random.Random(seed)
    selected = []
    for cls, need in counts_per_class.items():
        pool = available_indices_by_class.get(cls, [])
        if len(pool) < need:
            raise ValueError(f"Not enough samples in class '{cls}' (need {need}, have {len(pool)}).")
        chosen = rng.sample(pool, need)
        selected.extend(chosen)
        # Remove chosen from pool
        remaining = list(set(pool) - set(chosen))
        available_indices_by_class[cls] = remaining
    return selected

def load_dataset(num_train_samples, num_test_samples, num_public_samples, seed: int = 42):
    """
    Loads EuroSAT (RGB) and returns uniformly-sampled, non-overlapping train/test/public sets
    across all 10 classes, with consistent preprocessing and label handling.
    """
    # Use the official EuroSAT RGB split to ensure all 10 classes are present
    # (class names: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
    #  Pasture, PermanentCrop, Residential, River, SeaLake)
    full_dataset = hf_load_dataset("eurosat", "rgb", split="train[:100%]")

    # If coming from other variants with different column names, guard-rename
    if "image_path" in full_dataset.column_names and "image" not in full_dataset.column_names:
        full_dataset = full_dataset.rename_column("image_path", "image")
    if "class" in full_dataset.column_names and "label" not in full_dataset.column_names:
        full_dataset = full_dataset.rename_column("class", "label")

    # Ensure image column is the right type
    full_dataset = full_dataset.cast_column("image", datasets.Image())

    # Determine class label feature & class names
    if isinstance(full_dataset.features["label"], ClassLabel):
        class_label = full_dataset.features["label"]
        unique_classes = list(range(len(class_label.names)))  # numeric labels
        class_names = class_label.names
    else:
        # Fallback if labels are strings
        unique_classes = sorted(set(full_dataset["label"]))
        class_label = ClassLabel(names=unique_classes)
        class_names = class_label.names

    num_classes = len(unique_classes)

    # Build per-class index pools (keys must match label values used in dataset)
    class_to_indices = defaultdict(list)
    for idx, y in enumerate(full_dataset["label"]):
        class_to_indices[y].append(idx)

    # Make a mutable copy for non-overlapping sampling across splits
    available_indices_by_class = {k: v.copy() for k, v in class_to_indices.items()}

    # Compute how many to take per class for each split (uniform + fair remainder)
    train_counts = _compute_counts_per_class(num_train_samples, num_classes, unique_classes, seed=seed)
    test_counts  = _compute_counts_per_class(num_test_samples,  num_classes, unique_classes, seed=seed + 1)
    public_counts = _compute_counts_per_class(num_public_samples, num_classes, unique_classes, seed=seed + 2)

    # Sample non-overlapping indices
    train_indices  = _sample_indices_by_counts(available_indices_by_class, train_counts,  seed=seed)
    test_indices   = _sample_indices_by_counts(available_indices_by_class, test_counts,   seed=seed + 1)
    public_indices = _sample_indices_by_counts(available_indices_by_class, public_counts, seed=seed + 2)

    # Slice datasets
    train_slice  = full_dataset.select(train_indices)
    test_slice   = full_dataset.select(test_indices)
    public_slice = full_dataset.select(public_indices)

    # Prepare (resize, tensorize, label-encode if needed)
    train_data = prepare_dataset(train_slice, class_label)
    test_data  = prepare_dataset(test_slice,  class_label)
    public_train_data = prepare_dataset(public_slice, class_label)

    dataset = DatasetDict({
        "train": train_data,
        "test": test_data
    })

    public_data = DatasetDict({
        "train": public_train_data,
        "test": None
    })

    return dataset, num_classes, class_names, public_data
