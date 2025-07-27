import torch
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import json
from Config import args
from sklearn.metrics import accuracy_score
import gc
from torch.utils.data import DataLoader, TensorDataset



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
def Train2(model, data, optimizer, scheduler, loss_fn,  batch_size, epochs, device, debug):

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
        epoch_test_acc.append( Evaluate(model,  data["test"]["image"], data["test"]["label"], device)[0] )
        if debug: print("Epoch {}/{} ===> Loss: {:.2f}, Train accuracy: {:.2f}, Test accuracy: {:.2f}".format(epoch, epochs, epoch_loss[-1], epoch_acc[-1], epoch_test_acc[-1]))
    

    
    # Clean up DataLoader to free memory
    del dataset
    gc.collect()
    torch.cuda.empty_cache() # Only needed if you're using CUDA

    
    
    return epoch_loss, epoch_acc, epoch_test_acc

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
def Model_Size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_mb = total_params * 4 / (1024 ** 2)
    print(f"Number of trainable parameters: {total_params} | Total size: {total_size_mb:.2f} MB")


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





