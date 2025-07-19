import numpy as np
import datasets
import random
import torch
from Config import args

##############################################################################################################
def ddf(x):
    x = datasets.Dataset.from_dict(x)
    x.set_format("torch")
    return x
##############################################################################################################
def shuffling(a, b):
    return np.random.randint(0, a, b)
##############################################################################################################
def normalization(batch):
    normal_image = batch["image"] / 255
    return {"image": normal_image, "label": batch["label"]}
##############################################################################################################
def load_data_from_Huggingface():
    
    ## Loading MNIST Dataset
    if args.dataset in ["mnist", "MNIST"]:
        loaded_dataset = datasets.load_dataset("mnist", split=['train[:100%]', 'test[:100%]'])

    ## Loading CIFAR10 Dataset
    if args.dataset  in ["CIFAR10", "CIFAR-10", "cifar10", "cifar-10"]: loaded_dataset = datasets.load_dataset("cifar10", split=['train[:50%]', 'test[:50%]'])



    dataset = datasets.DatasetDict({   "train":ddf(loaded_dataset[0][shuffling(loaded_dataset[0].num_rows, args.num_train_samples)]),
                                        "test":ddf(loaded_dataset[1][shuffling(loaded_dataset[1].num_rows, args.num_test_samples)])
                                   })

    if not "image" in dataset["train"].column_names: dataset = dataset.rename_column(dataset["train"].column_names[0], 'image')
    if not "label" in dataset["train"].column_names: dataset = dataset.rename_column(dataset["train"].column_names[1], 'label')
    


    dataset.set_format("torch", columns=["image", "label"])
    if not dataset["train"]["image"].max() <= 1: 
        dataset = dataset.map(normalization, batched=True)
    name_classes = loaded_dataset[0].features["label"].names
    
    
    return dataset, len(name_classes), name_classes

##############################################################################################################
def data_distributing(centralized_data, num_classes):
    train_data = ddf(centralized_data['train'][:])
    test_data = centralized_data['test'][:]
    distributed_data = []
    samples = np.random.dirichlet(np.ones(num_classes)*args.alpha_dirichlet, size=args.num_clients)
    num_samples = np.array(samples*int(len(train_data)/args.num_clients))
    num_samples = num_samples.astype(int)


    available_data = train_data["label"]

    for i in range(args.num_clients):
        idx_for_client = []
        for c in range(num_classes):
            num = num_samples[i][c]
            
            
            if (available_data == c).sum().item() < num: num = (available_data == c).sum().item()
            
            if num == 0: 
                idx_per_class = np.random.choice( np.where(train_data["label"]==c)[0], 1 , replace=False)
                idx_for_client.extend( idx_per_class )
            else:
                idx_per_class = np.random.choice( np.where(available_data==c)[0], num , replace=False)
                idx_for_client.extend( idx_per_class )
                available_data[idx_per_class] = -1000            


            
        random.shuffle(idx_for_client)
        train_data_client = train_data[idx_for_client]
        client_data = datasets.DatasetDict({  "train": ddf(train_data_client),  "test": ddf(test_data)  })
        distributed_data.append(client_data)

    return distributed_data, num_samples
##############################################################################################################
