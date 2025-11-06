import numpy as np
import datasets
import torch
from datasets import load_dataset as hf_load_dataset, DatasetDict, ClassLabel
from collections import defaultdict
import random
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Any

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

def prepare_dataset(data, class_label: ClassLabel = None):
    # Ensure canonical column names
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    # Cast image to PIL-backed datasets.Image
    data = data.cast_column("image", datasets.Image())

    # Only encode labels if they're strings (if already ClassLabel/ints -> keep as is)
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
######################################################################################################

# The canonical 10 EuroSAT class names (for validation / nice ordering)
EUROSAT_10 = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

def _infer_columns(ds) -> Tuple[str, str]:
    """
    Try to find the image and label columns robustly.
    """
    # Common image column names
    img_candidates = ["image", "image_path", "img", "path", "file", "filepath"]
    # Common label column names
    label_candidates = ["label", "class", "Class", "category", "y", "target"]

    img_col = None
    for c in img_candidates:
        if c in ds.column_names:
            img_col = c
            break
    if img_col is None:
        # Fallback: assume first column is image
        img_col = ds.column_names[0]

    label_col = None
    for c in label_candidates:
        if c in ds.column_names:
            label_col = c
            break
    if label_col is None:
        # Fallback: assume second column is label
        label_col = ds.column_names[1] if len(ds.column_names) > 1 else ds.column_names[0]

    return img_col, label_col

def _compute_counts_per_class(total_samples: int, classes: List[Any], seed: int) -> Dict[Any, int]:
    """Uniform counts per class with fair remainder distribution."""
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
    rng = random.Random(seed)
    chosen_all = []
    for cls, need in counts.items():
        pool = available.get(cls, [])
        if len(pool) < need:
            raise ValueError(f"Not enough samples in class '{cls}': need {need}, have {len(pool)}")
        chosen = rng.sample(pool, need)
        chosen_all.extend(chosen)
        # remove chosen
        remaining = list(set(pool) - set(chosen))
        available[cls] = remaining
    return chosen_all

def _class_counts(y_list):
    counts = defaultdict(int)
    for y in y_list:
        counts[y] += 1
    return dict(counts)

def load_dataset(num_train_samples: int,
                 num_test_samples: int,
                 num_public_samples: int,
                 seed: int = 42,
                 require_all_10: bool = True):
    """
    Load 'mikewang/EuroSAT', then uniformly sample across *all present classes*,
    creating non-overlapping train/test/public splits.

    If `require_all_10=True`, raises with a helpful report if the dataset does not
    contain all 10 EuroSAT classes.
    """
    # Load the community dataset you were using
    full_dataset = hf_load_dataset("mikewang/EuroSAT", split="train[:100%]")

    # Robustly align columns
    img_col, label_col = _infer_columns(full_dataset)
    if img_col != "image":
        full_dataset = full_dataset.rename_column(img_col, "image")
    if label_col != "label":
        full_dataset = full_dataset.rename_column(label_col, "label")

    # Ensure images are typed as datasets.Image
    full_dataset = full_dataset.cast_column("image", datasets.Image())

    # Determine classes as they appear in the dataset
    labels_raw = full_dataset["label"]
    # If label feature is ClassLabel -> use ints; else strings
    label_feature = full_dataset.features.get("label", None)
    if isinstance(label_feature, ClassLabel):
        # numeric labels 0..K-1, with names in label_feature.names
        unique_classes = list(range(len(label_feature.names)))
        class_names = label_feature.names
        to_name = lambda x: class_names[x]
    else:
        unique_classes = sorted(set(labels_raw))
        class_names = unique_classes  # labels are already strings
        to_name = lambda x: x

    # Validate presence of all 10 classes (optional)
    present_names = sorted({to_name(c) for c in unique_classes})
    if require_all_10:
        missing = sorted(set(EUROSAT_10) - set(present_names))
        if missing:
            # Provide a helpful diagnostic
            counts_now = _class_counts(labels_raw)
            raise RuntimeError(
                "This dataset copy does not contain all 10 EuroSAT classes.\n"
                f"Present: {present_names}\n"
                f"Missing: {missing}\n"
                f"Per-class counts (raw): {counts_now}\n"
                "-> Either switch to the official 'eurosat' dataset (once available), "
                "or set require_all_10=False to sample uniformly over the classes that are present."
            )

    num_classes = len(unique_classes)

    # Build per-class index pools
    class_to_indices = defaultdict(list)
    for idx, y in enumerate(labels_raw):
        class_to_indices[y].append(idx)

    # Non-overlapping sampling across splits
    available = {k: v.copy() for k, v in class_to_indices.items()}

    train_counts  = _compute_counts_per_class(num_train_samples,  unique_classes, seed=seed)
    test_counts   = _compute_counts_per_class(num_test_samples,   unique_classes, seed=seed+1)
    public_counts = _compute_counts_per_class(num_public_samples, unique_classes, seed=seed+2)

    train_indices  = _sample_indices_by_counts(available, train_counts,  seed=seed)
    test_indices   = _sample_indices_by_counts(available, test_counts,   seed=seed+1)
    public_indices = _sample_indices_by_counts(available, public_counts, seed=seed+2)

    # Slice datasets
    train_slice  = full_dataset.select(train_indices)
    test_slice   = full_dataset.select(test_indices)
    public_slice = full_dataset.select(public_indices)

    # Build a stable label encoder (string names) for downstream consistency
    # If we had ClassLabel already, reuse it; otherwise make one from present class names.
    if isinstance(label_feature, ClassLabel):
        class_label = label_feature
        class_names_out = class_label.names
    else:
        # Use the canonical EuroSAT order if available; otherwise sorted present names
        ordered_names = [c for c in EUROSAT_10 if c in present_names]
        # Add any extra names not in canonical (unlikely) at the end
        ordered_names += [c for c in present_names if c not in ordered_names]
        class_label = ClassLabel(names=ordered_names)
        class_names_out = ordered_names

    # Prepare datasets (resize, to tensor, and encode if needed)
    train_data = prepare_dataset(train_slice,  class_label)
    test_data  = prepare_dataset(test_slice,   class_label)
    public_train_data = prepare_dataset(public_slice, class_label)

    dataset = DatasetDict({"train": train_data, "test": test_data})
    public_data = DatasetDict({"train": public_train_data, "test": None})

    return dataset, num_classes, class_names_out, public_data
