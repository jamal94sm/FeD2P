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
def Train(model, data, optimizer, scheduler, loss_fn,  batch_size, epochs, device, debug):

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
def check_if_softmax(x):
    # Check if the input is softmax probabilities
    device = x.device
    if torch.all((x >= 0) & (x <= 1)) and torch.allclose(x.sum(dim=1), torch.ones(x.size(0), device=device), atol=1e-6):
        return True
    else:  
        return False

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
##############################################################################################################
def adjust_temperature_orginal(inputs, iteration, optimal_temperature, is_softmax):
    def change_temperature(probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
        scaled_logits = torch.log(probabilities) / temperature
        adjusted_probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        return adjusted_probs

    def entropy(probabilities):
        ents = []
        for prob in probabilities:
            ents.append( sum([p*torch.log2(1/p) for p in prob]) )
        return torch.tensor(ents)

    def find_temperature(inputs, down_entropy, up_entropy):
        if is_softmax:
           inputs = torch.log(inputs)
        temps = torch.logspace(-2, 1, steps=50, device='cpu').to(inputs.device)
        for temp in temps:  # Temperature from 1e-2 to 1e1
            if temp==0: continue
            probabilities = torch.nn.functional.softmax(inputs / temp, dim=1)
            current_entropy = torch.mean(entropy(probabilities)) # Average entropy over the batch
            if ( down_entropy < current_entropy <  up_entropy ): 
                #print("found:", temp, down_entropy ,current_entropy ,up_entropy)
                return probabilities, temp

        return probabilities, temp  # Return the last temperature if convergence was not reached

    if iteration == 0:
        input_length = inputs.shape[-1]
        up_entropy = 0.99*torch.log2(torch.tensor(float(input_length), device=inputs.device))  # Entropy of a uniform distribution
        down_entropy = 0.95*torch.log2(torch.tensor(float(input_length), device=inputs.device))  # Entropy of a uniform distribution
        probabilities,  optimal_temperature = find_temperature(inputs, down_entropy, up_entropy)
    else:
        probabilities = torch.nn.functional.softmax(inputs / optimal_temperature, dim=1)
    return probabilities, optimal_temperature
############################################################################################################## 
def plot(arrays, names=[""], title=args.output_name, xlabel='rounds', ylabel='accuracy %', file_name=args.output_name):
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
def save_as_json(to_save, config, test_acc=None, file_name="", output_dir="results"):
    if isinstance(to_save, np.ndarray):
        to_save = to_save.tolist()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name + ".json")

    # Extract config attributes and convert torch.device to string
    config_dict = {
        key: str(value) if isinstance(value, torch.device) else value
        for key, value in vars(config).items()
    }

    # Add the main object to save
    config_dict["stored"] = to_save

    # Add test accuracy if provided
    if test_acc is not None:
        config_dict["test_acc"] = float(test_acc)

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
def run_in_parallel(clients):
    streams = [torch.cuda.Stream() for _ in clients]

    # Launch training in parallel CUDA streams
    for client, stream in zip(clients, streams):
        with torch.cuda.stream(stream):
            client.local_training()

    # Wait for all to finish
    torch.cuda.synchronize()  
        


##############################################################################################################






