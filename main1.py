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
from Config import args 
import time
import psutil










def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)
    transformers.set_seed(seed)


def clean_memory(FM, processor, tokenizer):
    # Free-up the memory 
    del FM
    del processor
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()    




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

    p_Model = MyModels.Prompt_Generator_plus_FM(FM, processor, tokenizer, num_classes, name_classes).to(device)
    server = MyPlayers.Server(p_Model, clients)





    # ===================== Zero-Shot Evaluation =====================
    if "zero_shot" in args.setup: #proto
        best_logits = server.zero_shot(Dataset["train"], FM, processor, tokenizer, prototype=True)
        accuracy = MyUtils.Evaluate2(
            ground_truth = Dataset["train"]["label"],
            output_logits = MyUtils.extend_proto_outputs_to_labels(Dataset, best_logits)
        )
        print(f"Accuracy of the teacher model: {accuracy}%\n")
        
    elif "bc" in args.setup:
        best_logits = [
            server.zero_shot(client.data["train"], FM, processor, tokenizer, prototype=False)
            for client in clients
        ]
        
        accuracy = [
            MyUtils.Evaluate2(
            ground_truth=client.data["train"]["label"],
            output_logits=best_logits[client.ID]
            )
            for client in clients
        ]
        print(f"Accuracy of the teacher model: {accuracy}%\n")





    
        


    # ===================== Training Rounds =====================


    for client in clients:
        client.local_training()



    for round_idx in range(args.rounds):
        print("=" * 20, f" Round {round_idx + 1}/{args.rounds} ", "=" * 20)
        
        for client in clients:
            client.cal_proto_logits()
        agg = server.aggregation()
        
        
        
        
        if "ft" in args.setup:
            print("-" * 20, "Server Distillation Phase")
            
            server.distill_generator(agg)
            general_knowledge = server.get_general_knowledge()
        


        
        for client in clients:
            print(f"Distillation process of client {client.ID}:")

            if "zero_shot" in args.setup:
                client.local_distillation(best_logits[client.ID], prototype=True)
            elif "FedMD" in args.setup:
                client.local_distillation(agg, prototype=True)
            elif "ft" in args.setup:
                client.local_distillation(general_knowledge, prototype=True)
            elif "local" in args.setup:
                client.local_training()
            else:
                raise ValueError("This is a custom error message.")






        
    # ===================== Save Results =====================
    avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    MyUtils.save_as_json(avg_test_Acc, args, file_name= args.output_name + "accuracy_"+args.setup)








        

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
    # ft: clip is fine-tuned --- mean: average of descriptions' embedding is used for refrence
    # M: multiple descriptions --- sift: only true_labeled soft labels be shared with the server
    configurations = [
        #{"setup": "local"},
        #{"setup": "FedMD"},
        #{"setup": "zero_shot"},
        {"setup": "ft_M_yn"},
        #{"setup": "ft_BN_sift_M_mean_yn"},
        #{"setup": "ft_M_yn"},
        #{"setup": "ft_yn"},
    ]

    for config in configurations:
        args.setup = config["setup"]
        
        
            
        separator = "=" * 40
        print(f"\n{separator} Running configuration: {args.setup} {separator}")
    
        
        ### Load the CLIP model for each setup 
        FM, processor, tokenizer = MyModels.load_clip_model()
        
        main()
        
        clean_memory(FM, processor, tokenizer)
        print(f"{separator} Simulation is over for configuration {args.setup} {separator}\n")







    
    
    
    # ===================== Data Loading and Plot =====================
    results_dir = "results"  # Directory containing your JSON files    
    stored_arrays = []  # Collect all 'stored' arrays
    names = []
    for file in os.listdir(results_dir):
        if file.endswith(".json") and file.startswith(args.output_name):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                if "stored" in data:
                    arr = np.array(data["stored"])
                    stored_arrays.append(arr) 
                if "setup" in data:
                    names.append(data["setup"])

    MyUtils.plot(stored_arrays, names)

    

    #MyUtils.play_alert_sound()
    







