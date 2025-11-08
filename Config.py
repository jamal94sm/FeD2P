
import argparse
import torch

'''
# ===================== Argument Parsing =====================
def get_args():
    parser = argparse.ArgumentParser(description="FedD2P")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu") # for runing on GPU
    #parser.add_argument('--device',default="mps" if torch.backends.mps.is_available() else "cpu") # for runing on mps MAC OS

    parser.add_argument('--setup', default="local")
    parser.add_argument('--output_name', type=str, default='_ResNet10_flowers_850_alpha10_')
    
    parser.add_argument('--local_model_name', type=str, default="ResNet10")
    parser.add_argument('--dataset', type=str, default="flowers") # fashion_mnist, cifar10, flowers, eurosat

    parser.add_argument('--num_clients', type=int, default= 5)
    parser.add_argument('--rounds', type=int, default=30)
    
    parser.add_argument('--num_train_samples', type=int, default=850)
    parser.add_argument('--num_test_samples', type=int, default=170)
    parser.add_argument('--num_public_samples', type=int, default=170)
    parser.add_argument('--num_synth_img_per_class', type=int, default=10)
    parser.add_argument('--alpha_dirichlet', type=float, default=100)
    
    parser.add_argument('--local_batch_size', type=int, default=16)
    parser.add_argument('--global_batch_size', type=int, default=16)
    parser.add_argument('--num_prompts', type=int, default=10)
    parser.add_argument('--global_epochs', type=int, default=5)
    parser.add_argument('--local_learning_rate', type=float, default=0.001)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--global_learning_rate', type=float, default=0.01)
    
    
    parser.add_argument('--Foundation_model', type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument('--default_temp', type=float, default=1)
    parser.add_argument('--load_saved_models', action='store_true')
    parser.add_argument('--generator_name', type=str, default="AttentionModel")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_template', type=str, default = "This is a photo of a {}")

    return parser.parse_args()


args = get_args()


'''


# ===================== Toy Example =====================
def get_args():
    parser = argparse.ArgumentParser(description="FedD2P")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu") # for runing on GPU
    #parser.add_argument('--device',default="mps" if torch.backends.mps.is_available() else "cpu") # for runing on mps MAC OS

    parser.add_argument('--setup', default="local")
    parser.add_argument('--output_name', type=str, default='_imagenette_')
    
    parser.add_argument('--num_clients', type=int, default= 2)
    parser.add_argument('--local_model_name', type=str, default="ResNet18")
    parser.add_argument('--dataset', type=str, default="imagenette") 
    
    parser.add_argument('--num_train_samples', type=int, default=100)
    parser.add_argument('--num_test_samples', type=int, default=50)
    parser.add_argument('--num_public_samples', type=int, default=50)
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--num_synth_img_per_class', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    
    parser.add_argument('--num_prompts', type=int, default=1)
    parser.add_argument('--global_epochs', type=int, default=1)
    parser.add_argument('--Foundation_model', type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument('--local_learning_rate', type=float, default=0.001)
    parser.add_argument('--local_batch_size', type=int, default=64)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--global_learning_rate', type=float, default=0.01)
    parser.add_argument('--global_batch_size', type=int, default=32)
    parser.add_argument('--default_temp', type=float, default=1)
    parser.add_argument('--load_saved_models', action='store_true')
    parser.add_argument('--generator_name', type=str, default="AttentionModel")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_template', type=str, default = "This is a photo of a {}")

    return parser.parse_args()

args = get_args()

