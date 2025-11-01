import torch
import transformers
import numpy as np
import random
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


from Config import get_args
args = get_args()

import MyUtils
import MyDatasets
import MyModels
import MyPlayers

import torchvision
import time
import json
import os
import gc
from Config import args 
import time
import psutil







##############################################################################################################
##############################################################################################################

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

##############################################################################################################
##############################################################################################################
def clean_memory(FM, processor, tokenizer):
    # Free-up the memory 
    del FM
    del processor
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()    


import gc
import torch

def clean_up_memory(*args):
    """
    Frees GPU and CPU memory by deleting passed objects, 
    collecting garbage, and clearing CUDA cache.
    
    Args:
        *args: Any large objects (e.g., models, data) to delete.
    """
    for obj in args:
        del obj
    gc.collect()
    torch.cuda.empty_cache()

##############################################################################################################
##############################################################################################################

def main():

    device = torch.device(args.device)
    print(f'Device: {device}')
    
    # ===================== Build public dataset =====================
    '''
    synth_img_dir = "/project/def-arashmoh/shahab33/FedPD/Synthetic_Image/EuroSAT"
    #public_data = MyUtils.load_synthetic_images( name_classes, data_dir=synth_img_dir, max_per_class=args.num_synth_img_per_class)
    if "sidclip" in args.setup or "open_vocab" in args.setup: 
        public_data = MyUtils.load_synthetic_images(name_classes, 
                                                image_size=distributed_dataset[0]["train"]["image"].shape[-2:], 
                                                data_dir=synth_img_dir, 
                                                max_per_class=2*args.num_synth_img_per_class)
    else: 
        public_data = MyUtils.load_synthetic_images(name_classes, 
                                                image_size=distributed_dataset[0]["train"]["image"].shape[-2:], 
                                                data_dir=synth_img_dir, 
                                                max_per_class=args.num_synth_img_per_class)
    
    id = args.num_clients-1
    last_client = MyPlayers.Device(id, distributed_dataset[id], num_classes, name_classes , None)
    public_data_2 = last_client.data
    '''

    public_data = synthetic_public_data
    public_data_2 = original_public_data

    # ===================== Client and Server Setup =====================
    clients = [ MyPlayers.Device( id, distributed_dataset[id], num_classes, name_classes, public_data ) for id in range(args.num_clients-1) ]




    p_Model = MyModels.Image_prompting_plus_Fm(FM, processor, tokenizer, num_classes, name_classes).to(device)
    server = MyPlayers.Server(p_Model, clients, public_data)


 

    # ===================== Zero-Shot Evaluation =====================
    if "zero_shot" in args.setup or "open_vocab" in args.setup or "fl_vocab" in args.setup:
        zero_shot_logits = server.zero_shot(
            public_data["train"], 
            FM,
            processor,
            tokenizer,
            proto = True if "proto" in args.setup else False,)




    # ==================================================================
    # ===================== Perfoming the main loop ====================
    # ==================================================================
    for round in range(args.rounds):
        print("=" * 20, f" Round {round + 1}/{args.rounds} ", "=" * 20)
        

        #==================================================================
        if 'local' in args.setup:
            for client in clients:
                client.local_selective_training(client.data)
                client.local_selective_training(public_data_2, eval=False)
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            continue
        #==================================================================
        elif 'fedavg' in args.setup:
            for client in clients:
                client.local_selective_training(client.data)
                client.local_selective_training(public_data_2, eval=False)
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            server.fedavg_aggregation_and_implanting()
            continue
        #==================================================================
        elif 'fedmd' in args.setup:
            for client in clients:
                
                if round > 0 :  
                    client.local_distillation(
                        public_data_2,
                        agg, 
                        proto = True if "proto" in args.setup else False,
                        )
                
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                
                client.cal_logits( 
                    public_data_2,
                    proto = True if "proto" in args.setup else False,
                    sifting = True if "sift" in args.setup else False,
                    )
            agg = server.aggregation()
            continue
        #==================================================================
        elif 'fedmd_synth' in args.setup:
            for client in clients:
                
                if round > 0 :  
                    client.local_distillation(
                        client.public_data,
                        agg, 
                        proto = True if "proto" in args.setup else False,
                        )
                
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                
                client.cal_logits( 
                    client.public_data,
                    proto = True if "proto" in args.setup else False,
                    sifting = True if "sift" in args.setup else False,
                    )
            agg = server.aggregation()
            continue
        #==================================================================
        elif 'zero_shot' in args.setup:
            for client in clients:
                client.local_training()
                client.local_distillation(
                    client.public_data,
                    zero_shot_logits, 
                    proto = True if "proto" in args.setup else False,
                    )
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            continue
        #==================================================================
        elif "proposed_real" in args.setup:
            for client in clients:
                
                if round > 0 :  
                    client.local_distillation(
                        public_data_2,
                        general_knowledge, 
                        proto = True if "proto" in args.setup else False,
                        )
                
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                
                client.cal_logits( 
                    public_data_2,
                    proto = True if "proto" in args.setup else False,
                    sifting = True if "sift" in args.setup else False,
                    )
            agg = server.aggregation()
            print("-" * 20, "Server Distillation Phase")
            server.distill_generator(server.public_data, agg)
            general_knowledge = server.get_general_knowledge()
            continue
        #==================================================================
        elif "proposed" in args.setup:
            for client in clients:
                
                if round > 0 :  
                    client.local_distillation(
                        client.public_data,
                        general_knowledge, 
                        proto = True if "proto" in args.setup else False,
                        )
                
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                
                client.cal_logits( 
                    client.public_data,
                    proto = True if "proto" in args.setup else False,
                    sifting = True if "sift" in args.setup else False,
                    )
            agg = server.aggregation()
            print("-" * 20, "Server Distillation Phase")
            server.distill_generator(server.public_data, agg)
            general_knowledge = server.get_general_knowledge()
            continue
        #==================================================================
        elif 'open_vocab' in args.setup:

            client = clients[0]
            client.local_distillation(
                client.public_data,
                zero_shot_logits, 
                proto = True if "proto" in args.setup else False,
                )
            client.test_Acc.append(MyUtils.Evaluate(client.model,  client.data["test"]["image"], client.data["test"]["label"], device)[0])
        #==================================================================
        elif 'fl_vocab' in args.setup:
            if round == 0:
                old_local_epochs = args.local_epochs
                args.local_epochs = args.rounds
                for client in clients:
                    client.local_distillation(
                        client.public_data,
                        zero_shot_logits, 
                        proto = True if "proto" in args.setup else False,
                        )
                args.local_epochs = old_local_epochs

            for client in clients:
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
            continue
        #==================================================================
        elif 'sidclip' in args.setup:
            client = clients[0]

            if round == 0:
                old_local_epochs = args.local_epochs
                args.local_epochs = args.rounds


                teacher_model = MyModels.clip_plus_linear_head(FM, processor, tokenizer, num_classes, name_classes, args.device).to(args.device)
                few_shot_data = MyUtils.get_few_shot_subset(client.data, num_shots=5)
                optimizer = torch.optim.Adam(teacher_model.parameters(), lr=args.local_learning_rate)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
                loss_fn = torch.nn.functional.cross_entropy
                MyUtils.Train(teacher_model, 
                            few_shot_data, 
                            optimizer, 
                            scheduler, 
                            loss_fn,  
                            batch_size = 8, 
                            epochs = 20,
                            device = args.device,
                            debug = False, 
                            eval = False)   
                teacher_logits = teacher_model.inference(client.public_data)


                client.local_distillation(
                    client.public_data,
                    teacher_logits, 
                    proto = True if "proto" in args.setup else False,
                    )
                
                del teacher_model
                torch.cuda.empty_cache()

                args.local_epochs = old_local_epochs

                client.test_Acc.append( MyUtils.Evaluate(client.model,  client.data["test"]["image"], client.data["test"]["label"], device)[0] )
        #=================================================================
        elif "koala" in args.setup: 
            for client in clients:
                client.local_training()
                print(f'Client: {client.ID:<10} train_acc: {client.Acc[-1]:<8.2f} test_acc: {client.test_Acc[-1]:<8.2f}')
                if round > 0 :  
                    client.local_distillation(
                        public_data_2,
                        general_knowledge, 
                        proto = True if "proto" in args.setup else False,
                        )

            global_model = server.fedavg_aggregation()
            train_loader = torch.utils.data.DataLoader(public_data_2['train'], batch_size=32)
            global_model.eval()
            global_logits = torch.cat([ global_model(batch['image'].to(device)).cpu() for batch in torch.utils.data.DataLoader(public_data_2['train'], batch_size=32) ], dim=0)
            server.distill_generator(public_data_2, global_logits)
            general_knowledge = server.get_general_knowledge()
            continue
            




    # ===================== Save Results =====================
    if  args.setup == 'open_vocab' or args.setup == 'sidclip':
        avg_test_Acc = clients[0].test_Acc
    else:
        avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    MyUtils.save_as_json(avg_test_Acc, args, file_name= args.output_name + "accuracy_"+args.setup)

    

##############################################################################################################
##############################################################################################################

dataset_loaders = {
    "cifar10": MyDatasets.load_cifar10,
    "eurosat": MyDatasets.load_eurosat,
    "svhn": MyDatasets.load_svhn,
    "fashion_mnist": MyDatasets.load_fashion_mnist
}



if __name__ == "__main__":
    set_seed(42)
    
    loader = dataset_loaders.get(args.dataset.lower())
    if loader is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    Dataset, num_classes, name_classes, original_public_data = loader(args.num_train_samples, args.num_test_samples)


    distributed_dataset, num_samples = MyDatasets.data_distributing(Dataset, num_classes, args.alpha_dirichlet, args.num_clients)
    print("\n data distribution of devices: \n", num_samples)


    synthetic_public_data = MyUtils.load_synthetic_images(name_classes, 
                                                  image_size = dataset["train"]["image"][0].shape[-2:], 
                                                  data_dir = "/project/def-arashmoh/shahab33/FedPD/Synthetic_Image/EuroSAT",
                                                  max_per_class=args.num_synth_img_per_class)



    # ===================== Run for each configuration =====================
    # ft: clip is fine-tuned --- mean: average of descriptions' embedding is used for refrence
    # M: multiple descriptions --- sift: only true_labeled soft labels are shared with the server
    configurations = [
        {"setup": "proposed_yn"},
        {"setup": "fedmd_yn"},
        {"setup": "local"},
        {"setup": "fedavg"},
        {"setup": "proposed_real_yn"},
        #{"setup": "fedmd_synth_yn"},
        #{"setup": "zero_shot"},
        #{"setup": "open_vocab"},
        #{"setup": "koala"},
        #{"setup": "fl_vocab"},
        #{"setup": "sidclip"}
    ]


    for config in configurations:

        args.setup = config["setup"]
        separator = "=" * 40
        print(f"\n{separator} Running configuration: {args.setup} {separator}")
    
        ### Load the CLIP model for each setup 
        FM, processor, tokenizer = MyModels.load_clip_model()

        main()

        print(f"{separator} Simulation is over for configuration {args.setup} {separator}\n")

        
        clean_memory(FM, processor, tokenizer)
        








    

    
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
    







