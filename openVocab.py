import torch
import transformers
import numpy as np
import random
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import MyDatasets
import MyModels
import MyPlayers
import MyUtils
import torchvision
import time
import json
import os
import gc
from sklearn.metrics import accuracy_score
from Config import args 
import time
import psutil





















def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    tf.random.set_seed(seed)


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


separator = "=" * 40



##################################################################################
##################################################################################
def main():

    
    device = torch.device(args.device)
    print(f'Device: {device}')




    # ===================== Client and Server Setup =====================
    clients = [
        MyPlayers.Device(
            id,
            distributed_dataset[id],
            num_classes,
            name_classes
        ) for id in range(args.num_clients)
    ]

    p_Model = MyModels.Decoder_plus_FM(FM, processor, tokenizer, num_classes, name_classes).to(device)
    server = MyPlayers.Server(p_Model, clients)












    
    server.train_decoder(num_classes)
    out = server.get_generated_images()    
    out = out.permute(0, 2, 3, 1)
    
    
    
    # save the first image for fun
    image = out[0]
    image = (image - image.min()) / (image.max() - image.min())
    image = image.detach().cpu().numpy()
    plt.imsave("image0.png", image)  
    
    
    torch.save(server.zs, "latent_tensors.pt")







        








##################################################################################
##################################################################################
if __name__ == "__main__":
    
    
    set_seed(42)



    # ===================== Dataset and Model Loading =====================
    Dataset, num_classes, name_classes = MyDatasets.load_data_from_Huggingface()



    # ===================== Data Distribution =====================
    distributed_dataset, num_samples = MyDatasets.data_distributing(Dataset, num_classes)
    print("\n ]data distribution of devices: \n", num_samples)



    # ===================== Run for each configuration =====================
    configurations = [
        {"setup": "ft_yn"},
    ]



    for config in configurations:
        args.setup = config["setup"]
        print(f"\n{separator} Running configuration: {args.setup} {separator}")
    
        
        ### Load the CLIP model for each setup 
        FM, processor, tokenizer = MyModels.load_clip_model()
        
        main()


        clean_memory()
        print(f"{separator} Simulation is over for configuration {args.setup} {separator}\n")




