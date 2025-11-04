import torch
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import json
from sklearn.metrics import accuracy_score
import gc
from torch.utils.data import DataLoader, TensorDataset

from Config import args
############################################################################################################## 
############################################################################################################## 

def plot(arrays, names=[""], title='Comparison of Arrays', xlabel='rounds', ylabel='accuracy %', file_name="figure"):
    # Convert to numpy array with dtype=object to handle inhomogeneous sequences
    arrays = np.array(arrays, dtype=object)

    # Ensure names list matches the number of arrays
    if len(arrays) != len(names):
        names += [""] * abs(len(arrays) - len(names))

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    for arr, name in zip(arrays, names):
        arr = np.array(arr)  # Convert each individual array to numpy array
        plt.plot(arr, label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}.png")
    plt.show()

############################################################################################################## 
##############################################################################################################

def play_alert_sound():
    system = platform.system()
    if system == "Windows":
        import winsound
        duration = 1000  # milliseconds
        freq = 750  # Hz
        winsound.Beep(freq, duration)
    elif system == "Darwin":  # macOS
        os.system('say "Results are ready Ka Jamal delan"')
    else:  # Linux and others
        print('\a')  # ASCII Bell character

##############################################################################################################
############################################################################################################## 

def save_as_json(to_save, config, file_name="", output_dir="results"):
    
    if isinstance(to_save, np.ndarray):
        to_save = to_save.tolist()


    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name + ".json")

    # Extract config attributes and convert torch.device to string
    config_dict = {
        key: str(value) if isinstance(value, torch.device) else value
        for key, value in vars(config).items()
    }

    # Add the object to save
    config_dict["stored"] = to_save

    # Save to compact JSON
    with open(output_path, "w") as f:
        json.dump(config_dict, f, separators=(',', ':'))

    print(f"Data saved to {output_path}")

##############################################################################################################
##############################################################################################################

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

def estimate_flops(model, dataset_dict, criterion, optimizer, device, batch_size=4):
    model.to(device)
    model.train()


    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, hf_data):
            self.data = hf_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]['image']
            label = self.data[idx]['label']
            return image, label

    dataloader = DataLoader(TensorDataset(dataset_dict['train']), batch_size=batch_size)

    # Get input shape from one batch
    inputs, _ = next(iter(dataloader))
    input_size = tuple(inputs.shape)

    # Estimate FLOPs using torchinfo
    info = summary(model, input_size=input_size, verbose=0)
    flops_per_forward = 2 * info.total_mult_adds

    num_batches = len(dataloader)
    flops_per_batch = flops_per_forward * 2  # forward + backward
    total_flops = flops_per_batch * num_batches

    total_gflops = total_flops / 1e9  # Convert to GFLOPs
    #print("====> Total FLOPs: {:.2f} GFLOPs".format(total_gflops))
    return total_gflops


##############################################################################################################
##############################################################################################################

import torch
import numpy as np

def get_dataset_size_in_gb(dataset_dict):
    def estimate_split_size(split):
        total_bytes = 0
        for sample in split:
            image = sample['image']
            label = sample['label']
            if isinstance(image, torch.Tensor):
                total_bytes += image.numel() * image.element_size()
            elif isinstance(image, np.ndarray):
                total_bytes += image.nbytes
            elif hasattr(image, 'tobytes'):
                total_bytes += len(image.tobytes())
            total_bytes += 4  # Assuming label is int32
        return total_bytes

    total_bytes = 0
    if 'train' in dataset_dict:
        total_bytes += estimate_split_size(dataset_dict['train'])
    if 'test' in dataset_dict and dataset_dict['test'] is not None:
        total_bytes += estimate_split_size(dataset_dict['test'])

    size_gb = total_bytes / (1024 ** 3)
    return size_gb


##############################################################################################################
##############################################################################################################
def get_logits_size_in_gb(num_samples, num_classes):
    total_elements = num_samples * num_classes
    total_bytes = total_elements * 4  # float32 = 4 bytes
    size_gb = total_bytes / (1024 ** 3)
    return size_gb


##############################################################################################################
##############################################################################################################
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from datasets import Dataset, DatasetDict

def load_synthetic_images(class_names, image_size, data_dir, max_per_class=100):
    # Define transform to match CIFAR-10 format
    transform = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
    ])

    image_tensors = []
    label_tensors = []

    # Replace "T-shirt/top" with "T-shirt" if pre
    class_names = ["T-shirt" if name == "T-shirt / top" else name for name in class_names]

    class_counts = {class_name: 0 for class_name in class_names}

    # Walk through subdirectories
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                for class_name in class_names:
                    # Match class by folder name and check max limit
                    if class_name.lower() in root.lower() and class_counts[class_name] < max_per_class:
                        label = class_names.index(class_name)
                        image_path = os.path.join(root, filename)
                        try:
                            image = Image.open(image_path).convert("RGB")
                            tensor_image = transform(image)
                            image_tensors.append(tensor_image)
                            label_tensors.append(label)
                            class_counts[class_name] += 1
                        except Exception as e:
                            print(f"⚠️ Skipping {image_path}: {e}")
                        break  # stop checking other classes once matched

    # Check if we loaded anything
    if not image_tensors:
        raise ValueError(f"No images loaded from {data_dir}. "
                         f"Check folder structure and class_names={class_names}")

    print("✅ Loaded per class:", class_counts)

    train_images = torch.stack(image_tensors)
    train_labels = torch.tensor(label_tensors)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict({
        "image": train_images,
        "label": train_labels,
    })
    train_dataset.set_format("torch")

    return DatasetDict({
        "train": train_dataset,
        "test": None
    })

##############################################################################################################
##############################################################################################################

from datasets import DatasetDict
import random

def get_few_shot_subset(dataset_dict, num_shots=5):
    train_dataset = dataset_dict["train"]
    label_to_indices = {}

    # Group indices by label
    for idx, label in enumerate(train_dataset["label"]):
        label_to_indices.setdefault(label, []).append(idx)

    # Sample up to num_shots examples per class
    few_shot_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) == 0:
            continue
        sampled = indices if len(indices) < num_shots else random.sample(indices, num_shots)
        few_shot_indices.extend(sampled)

    # Select the few-shot subset
    few_shot_train = train_dataset.select(few_shot_indices)

    # Return as DatasetDict
    return DatasetDict({
        "train": few_shot_train,
        "test": None
    })

##############################################################################################################
##############################################################################################################

def clean_up_memory(*args):
    for obj in args:
        del obj
    gc.collect()
    torch.cuda.empty_cache()

##############################################################################################################
##############################################################################################################

import os
import json
import numpy as np

def load_and_plot_results(results_dir, output_name):

    stored_arrays = []
    names = []

    for file in os.listdir(results_dir):
        if file.endswith(".json") and file.startswith(output_name):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                if "stored" in data:
                    stored_arrays.append(np.array(data["stored"]))
                if "setup" in data:
                    names.append(data["setup"])

    plot(stored_arrays, names)

##############################################################################################################
##############################################################################################################

def model_size(model): # returns if Giga Bytes
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = total_params * 4  # 4 bytes per parameter (float32)
    size_gB = size_bytes / (1024 ** 3)
    return size_gB

##############################################################################################################     
##############################################################################################################

##############################################################################################################
def extend_proto_outputs_to_labels(input_data, proto_outputs):
    num_data = input_data["train"]["image"].shape[0]
    num_classes = len(  sorted(set(input_data["train"]["label"].tolist()))  )
    labels = input_data["train"]["label"]
    extended_outputs = torch.zeros(num_data, num_classes)
    for i in range(num_data):
        extended_outputs[i] = proto_outputs[labels[i].item()]
    return extended_outputs


##############################################################################################################
def FSL_data_preparing(samples, labels, num_shots): #Few-Shot Learning data preparing
    labels = labels.detach().numpy()
    samples = samples.detach().numpy()
    classes = list(set(labels))
    new_samples = []
    new_labels = []
    for cls in classes :
        ins = np.where(np.array(labels)==cls)[0]
        ins = ins[ : min(num_shots, len(ins))]
        for i in ins:
            new_samples.append(samples[i])
            new_labels.append(cls)
    return  torch.tensor(new_samples),  torch.tensor(new_labels)
##############################################################################################################
def plot(arrays, names=[""], title=args.output_name, xlabel='rounds', ylabel='accuracy %', file_name="figure"+args.output_name):
    # Convert to numpy array with dtype=object to handle inhomogeneous sequences
    arrays = np.array(arrays, dtype=object)

    # Ensure names list matches the number of arrays
    if len(arrays) != len(names):
        names += [""] * abs(len(arrays) - len(names))

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    for arr, name in zip(arrays, names):
        arr = np.array(arr)  # Convert each individual array to numpy array
        plt.plot(arr, label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}.png")
    plt.show()
##############################################################################################################
def check_if_softmax(x):
    # Check if the input is softmax probabilities
    device = x.device
    if torch.all((x >= 0) & (x <= 1)) and torch.allclose(x.sum(dim=1), torch.ones(x.size(0), device=device), atol=1e-6):
        return True
    else:  
        return False
##############################################################################################################
def Distil(model, extended_data, data, optimizer, scheduler, loss_fn, batch_size, epochs, device, debug):
    
    dataset = torch.utils.data.DataLoader(
        extended_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,              # Enables multi-processing
        pin_memory=True,            # Speeds up host-to-GPU transfer
        prefetch_factor=2,          # Controls preloading per worker
        persistent_workers=False    # Keeps workers alive between epochs
    )

    #dataset = torch.utils.data.DataLoader(extended_data, batch_size=batch_size, shuffle=True, drop_last=True)
    epoch_loss = []
    epoch_acc = []
    epoch_test_acc = []
    optimal_temp_teacher = 1
    optimal_temp_student = 1
    softness = check_if_softmax(extended_data["teacher_knowledge"][0:10])

    for epoch in range(epochs):
        batch_loss = []
        model.train()
        for batch in dataset:
            optimizer.zero_grad()
            pred = model(batch['student_model_input'].to(device))
            error1 = torch.nn.functional.cross_entropy(pred, batch["student_model_output"].to(device))

            if args.setup == "local":
                error2 = 0

            Sta = True if args.setup[-2] == "y" else False
            Tta = True if args.setup[-1] == "y" else False

            # if data == None:
            #     Sta = True
            #     Tta = False #KD

            if Sta:
                s, optimal_temp_student = adjust_temperature(pred, epoch, optimal_temp_student, is_softmax=False)
            else:
                s = torch.nn.functional.log_softmax(pred / args.default_temp, dim=-1)

            if Tta:
                t, optimal_temp_teacher = adjust_temperature(
                    batch["teacher_knowledge"].to(device),
                    epoch,
                    optimal_temp_teacher,
                    is_softmax=softness,
                )
            else:
                t = torch.nn.functional.softmax(batch["teacher_knowledge"].to(device) / args.default_temp, dim=-1)

            if Tta and Sta:
                error2 = (((optimal_temp_student + optimal_temp_teacher) / 2) ** 2) * torch.nn.KLDivLoss(
                    reduction='batchmean')(s.log(), t)
            elif not (Tta and Sta):
                error2 = (args.default_temp ** 2) * torch.nn.KLDivLoss(reduction="batchmean")(s, t)
            elif Tta and not Sta:
                error2 = (optimal_temp_teacher ** 2) * torch.nn.KLDivLoss(reduction='batchmean')(s.log(), t)
            elif not Tta and Sta:
                error2 = (optimal_temp_student ** 2) * torch.nn.KLDivLoss(reduction='batchmean')(s.log(), t)

            error = error1 + error2
            error.backward()
            optimizer.step()
            batch_loss.append(float(error))

        scheduler.step()
        epoch_loss.append(np.mean(batch_loss))

        if data:
            epoch_acc.append(Evaluate(model, data["train"]["image"], data["train"]["label"], device)[0])
            epoch_test_acc.append(Evaluate(model, data["test"]["image"], data["test"]["label"], device)[0])
            if debug:
                print("Epoch {}/{} ===> Loss: {:.2f}, Train accuracy: {:.2f}, Test accuracy: {:.2f}".format(
                    epoch, epochs, epoch_loss[-1], epoch_acc[-1], epoch_test_acc[-1]))
        else:
            if debug:
                print("Epoch {}/{} ===> Loss: {:.2f}".format(epoch, epochs, epoch_loss[-1]))

    # Clean up DataLoader to free memory
    del dataset
    gc.collect()
    torch.cuda.empty_cache() # Only needed if you're using CUDA

    

    return epoch_loss, epoch_acc, epoch_test_acc

##############################################################################################################
##############################################################################################################
def Evaluate(model, images, labels, device, batch_size=64):
    model.eval()
    correct = 0
    all_preds = []

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            pred = model(batch_images)
            predicted_classes = torch.argmax(pred, dim=1)
            correct += (predicted_classes == batch_labels).sum().item()
            all_preds.append(pred.cpu())

    accuracy = 100.0 * correct / len(labels)
    return accuracy, torch.cat(all_preds, dim=0)

##############################################################################################################
def Evaluate2(ground_truth, output_logits):
    with torch.no_grad():
        predicted_classes = torch.argmax(output_logits, dim=1)
        accuracy = accuracy_score(
            ground_truth.cpu().numpy(),
            predicted_classes.cpu().numpy()
        )
    return accuracy
##############################################################################################################
def Train(model, data, optimizer, scheduler, loss_fn,  batch_size, epochs, device, debug, eval):

    dataset = torch.utils.data.DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )


    epoch_loss = []
    epoch_acc = []
    epoch_test_acc = []
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for batch in dataset:
            optimizer.zero_grad()
            pred = model( batch['image'].to(device) )
            error = loss_fn(pred, batch["label"].to(device))
            error.backward()
            optimizer.step()
            batch_loss.append(float(error))
        scheduler.step()
        epoch_loss.append(np.mean(batch_loss))
        epoch_acc.append( Evaluate(model,  data["train"]["image"], data["train"]["label"], device)[0] )

        if data['test'] is not None and eval is True:
            epoch_test_acc.append( Evaluate(model,  data["test"]["image"], data["test"]["label"], device)[0] )
        
        if debug: print("Epoch {}/{} ===> Loss: {:.2f}, Train accuracy: {:.2f}, Test accuracy: {:.2f}".format(epoch, epochs, epoch_loss[-1], epoch_acc[-1], epoch_test_acc[-1]))
    

    
    # Clean up DataLoader to free memory
    del dataset
    gc.collect()
    torch.cuda.empty_cache() # Only needed if you're using CUDA

    
    
    return epoch_loss, epoch_acc, epoch_test_acc
##############################################################################################################
def Just_Train(model, data, optimizer, scheduler, loss_fn,  batch_size, epochs, device, debug):

    dataset = torch.utils.data.DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )


    epoch_loss = []
    epoch_acc = []
    epoch_test_acc = []
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for batch in dataset:
            optimizer.zero_grad()
            pred = model( batch['image'].to(device) )
            error = loss_fn(pred, batch["label"].to(device))
            error.backward()
            optimizer.step()
            batch_loss.append(float(error))
        scheduler.step()
        epoch_loss.append(np.mean(batch_loss))

    
    # Clean up DataLoader to free memory
    del dataset
    gc.collect()
    torch.cuda.empty_cache() # Only needed if you're using CUDA

    
    
    return epoch_loss, epoch_acc, epoch_test_acc

##############################################################################################################
def adjust_temperature(inputs, iteration, optimal_temperature, is_softmax, batch_size=512):
    def change_temperature(probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
        scaled_logits = torch.log(probabilities) / temperature
        adjusted_probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        return adjusted_probs

    def entropy(probabilities):
        # Compute entropy in batches to save memory
        ents = []
        with torch.no_grad():
            for i in range(0, probabilities.size(0), batch_size):
                batch = probabilities[i:i+batch_size]
                batch_entropy = -torch.sum(batch * torch.log2(batch + 1e-12), dim=1)
                ents.append(batch_entropy)
        return torch.cat(ents)

    def find_temperature(inputs, down_entropy, up_entropy):
        if is_softmax:
            inputs = torch.log(inputs + 1e-12)

        temps = torch.logspace(-2, 1, steps=50, device='cpu').to(inputs.device)
        last_probs = None
        for temp in temps:
            probs = torch.nn.functional.softmax(inputs / temp, dim=1)
            current_entropy = torch.mean(entropy(probs))
            last_probs = probs
            if down_entropy < current_entropy < up_entropy:
                return probs, temp
        return last_probs, temp

    with torch.no_grad():
        if iteration == 0:
            input_length = inputs.shape[-1]
            log2_input_len = torch.log2(torch.tensor(float(input_length), device=inputs.device))
            up_entropy = 0.99 * log2_input_len
            down_entropy = 0.95 * log2_input_len
            probabilities, optimal_temperature = find_temperature(inputs, down_entropy, up_entropy)
        else:
            probabilities = torch.nn.functional.softmax(inputs / optimal_temperature, dim=1)

    return probabilities, optimal_temperature
