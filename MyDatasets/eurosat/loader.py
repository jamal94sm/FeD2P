import numpy as np
import datasets
import torch
from datasets import load_dataset as hf_load_dataset, DatasetDict, ClassLabel
from collections import defaultdict
import random
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Any, Optional


######################################################################################################
# Optional helpers (kept for compatibility with your original code)
######################################################################################################

def ddf(x):
    x = datasets.Dataset.from_dict(x)
    x.set_format("torch")
    return x

def shuffling(a, b):
    return np.random.randint(0, a, b)


######################################################################################################
# Image preprocessing
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
# Dataset preparation (label encoding + preprocessing)
######################################################################################################

def prepare_dataset(data, class_label: Optional[ClassLabel] = None):
    # Ensure canonical column names (imagefolder already uses "image" and "label")
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    # Cast image column to PIL-backed datasets.Image
    data = data.cast_column("image", datasets.Image())

    # Encode labels to int if they're strings; if already ClassLabel/ints, skip
    if class_label is not None:
        label_feature = data.features.get("label", None)
        needs_encoding = True
        if isinstance(label_feature, ClassLabel):
            needs_encoding = False
        else:
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
# Uniform sampling utilities
######################################################################################################

# Canonical EuroSAT 10 class names
EUROSAT_10 = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

def _compute_counts_per_class(total_samples: int, classes: List[Any], seed: int) -> Dict[Any, int]:
    """
    Uniform counts per class with fair remainder distribution.
    If total_samples < len(classes), some classes will naturally get 0.
    """
    rng = random.Random(seed)
    base = total_samples // len(classes)
    rem = total_samples % len(classes)
    order = classes.copy()
    rng.shuffle(order)
    counts = {c: base for c in classes}
    for c in order[:rem]:
        counts[c] += 1
    return counts

def _sample_indices_by_counts(available: Dict[Any, List[int]], counts: Dict[Any, int], seed: int) -> List[int]:
    """
    Sample requested number of indices per class from available pools,
    removing sampled indices to prevent overlap across splits.
    """
    rng = random.Random(seed)
    chosen_all = []
    for cls, need in counts.items():
        pool = available.get(cls, [])
        if need > 0:
            if len(pool) < need:
                raise ValueError(f"Not enough samples in class '{cls}': need {need}, have {len(pool)}")
            chosen = rng.sample(pool, need)
            chosen_all.extend(chosen)
            # remove chosen
            remaining = list(set(pool) - set(chosen))
            available[cls] = remaining
    return chosen_all


######################################################################################################
# Main loader using local imagefolder
######################################################################################################

def load_dataset(num_train_samples: int,
                 num_test_samples: int,
                 num_public_samples: int,
                 seed: int = 42,
                 data_dir: str = "/home/shahab33/EuroSAT_RGB",
                 require_all_10: bool = True):
    """
    Create uniformly sampled, non-overlapping train/test/public splits from local EuroSAT (RGB) folder.

    Parameters
    ----------
    num_train_samples : int
        Total number of training samples to draw uniformly across classes.
    num_test_samples : int
        Total number of testing samples to draw uniformly across classes.
    num_public_samples : int
        Total number of public-train samples to draw uniformly across classes.
    seed : int
        Base seed for deterministic sampling; test/public use seed+1/seed+2.
    data_dir : str
        Path to local EuroSAT RGB directory with 10 class subfolders.
    require_all_10 : bool
        If True, enforce the presence of all 10 canonical classes.

    Returns
    -------
    dataset : datasets.DatasetDict
        Contains "train" and "test" splits (processed tensors).
    num_classes : int
        Number of unique classes present.
    class_names_out : List[str]
        Ordered class names used for label encoding.
    public_data : datasets.DatasetDict
        Contains "train" split (processed tensors) for public usage; "test" is None.
    """
    # Load entire folder as a single split (all items)
    full_dataset = hf_load_dataset("imagefolder", data_dir=data_dir, split="train[:100%]")

    # Ensure image is Image type
    full_dataset = full_dataset.cast_column("image", datasets.Image())

    # Determine classes (string or ClassLabel ints)
    labels_raw = full_dataset["label"]
    label_feature = full_dataset.features.get("label", None)
    if isinstance(label_feature, ClassLabel):
        unique_classes = list(range(len(label_feature.names)))  # numeric labels 0..K-1
        class_names = label_feature.names
        to_name = lambda x: class_names[x]
    else:
        unique_classes = sorted(set(labels_raw))                # string labels
        class_names = unique_classes
        to_name = lambda x: x

    # Validate presence of all 10 classes if requested
    present_names = sorted({to_name(c) for c in unique_classes})
    if require_all_10:
        missing = sorted(set(EUROSAT_10) - set(present_names))
        if missing:
            # Provide diagnostic and fail fast
            counts_now = {name: 0 for name in present_names}
            for y in labels_raw:
                counts_now[to_name(y)] += 1
            raise RuntimeError(
                "This dataset copy does not contain all 10 EuroSAT classes.\n"
                f"Present: {present_names}\n"
                f"Missing: {missing}\n"
                f"Per-class counts (raw): {counts_now}\n"
                "-> Ensure your EuroSAT_RGB directory has all 10 class subfolders.\n"
            )

    num_classes = len(unique_classes)

    # Group indices by class (keys must match the label values)
    class_to_indices = defaultdict(list)
    for idx, y in enumerate(labels_raw):
        class_to_indices[y].append(idx)

    # Copy pools for non-overlapping splits
    available = {k: v.copy() for k, v in class_to_indices.items()}

    # Compute per-class counts (train/test/public), fair remainder distribution
    train_counts  = _compute_counts_per_class(num_train_samples,  unique_classes, seed=seed)
    test_counts   = _compute_counts_per_class(num_test_samples,   unique_classes, seed=seed + 1)
    public_counts = _compute_counts_per_class(num_public_samples, unique_classes, seed=seed + 2)

    # Sample indices without overlap
    train_indices  = _sample_indices_by_counts(available, train_counts,  seed=seed)
    test_indices   = _sample_indices_by_counts(available, test_counts,   seed=seed + 1)
    public_indices = _sample_indices_by_counts(available, public_counts, seed=seed + 2)

    # Slice datasets
    train_slice  = full_dataset.select(train_indices)
    test_slice   = full_dataset.select(test_indices)
    public_slice = full_dataset.select(public_indices)

    # Build a stable ClassLabel for downstream consistency
    if isinstance(label_feature, ClassLabel):
        class_label = label_feature
        class_names_out = class_label.names
    else:
        # Prefer canonical EuroSAT order; append any extras (unlikely)
        ordered_names = [c for c in EUROSAT_10 if c in present_names]
        ordered_names += [c for c in present_names if c not in ordered_names]
        class_label = ClassLabel(names=ordered_names)
        class_names_out = ordered_names

    # Prepare datasets (resize, to tensor, and encode labels if needed)
    train_data = prepare_dataset(train_slice,  class_label)
    test_data  = prepare_dataset(test_slice,   class_label)
    public_train_data = prepare_dataset(public_slice, class_label)

    dataset = DatasetDict({"train": train_data, "test": test_data})
    public_data = DatasetDict({"train": public_train_data, "test": None})

    return dataset, num_classes, class_names_out, public_data

