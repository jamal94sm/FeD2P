
import numpy as np
import random
import datasets


##############################################################################################################
##############################################################################################################
def ddf(x):
    x = datasets.Dataset.from_dict(x)
    x.set_format("torch")
    return x


##############################################################################################################
##############################################################################################################

def data_distributing(centralized_data, num_classes, alpha_dirichlet, num_clients):
    train_data = ddf(centralized_data['train'][:])
    test_data = centralized_data['test'][:]
    distributed_data = []
    samples = np.random.dirichlet(np.ones(num_classes)*alpha_dirichlet, size=num_clients)
    num_samples = np.array(samples*int(len(train_data)/num_clients))
    num_samples = num_samples.astype(int)


    available_data = train_data["label"]

    for i in range(num_clients):
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
##############################################################################################################





