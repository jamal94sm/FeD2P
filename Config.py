
import argparse
import torch




# ===================== Argument Parsing =====================
def get_args():
    parser = argparse.ArgumentParser(description="Federated Learning with Prompt Tuning")

    parser.add_argument('--output_name', default="01")
    parser.add_argument('--setup', default="local")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu") 
    #parser.add_argument('--device',default="mps" if torch.backends.mps.is_available() else "cpu")# for runing on mps MAC OS
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--local_model_name', type=str, default="ResNet20") # EfficientNet, ResNet20, ResNet18, ResNet10, MobileNetV2
    parser.add_argument('--num_train_samples', type=int, default=50_000)
    parser.add_argument('--num_test_samples', type=int, default=3_000)
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--Foundation_model', type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--local_learning_rate', type=float, default=0.001)
    parser.add_argument('--local_batch_size', type=int, default=64)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--global_learning_rate', type=float, default=0.01)
    parser.add_argument('--global_batch_size', type=int, default=32)
    parser.add_argument('--global_epochs', type=int, default=50)
    parser.add_argument('--default_temp', type=float, default=1)
    parser.add_argument('--alpha_dirichlet', type=float, default=100)
    parser.add_argument('--load_saved_models', action='store_true')
    parser.add_argument('--generator_name', type=str, default="AttentionModel")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_template', type=str, default = "This is a photo of a {}")

    return parser.parse_args()

args = get_args()




























# An example command on terminal

# python main.py \
#   --setup kd \
#   --device cuda \
#   --num_clients 1 \
#   --local_model_name CNN \
#   --num_train_samples 3000 \
#   --num_test_samples 500 \
#   --dataset cifar10 \
#   --Foundation_model openai/clip-vit-base-patch32 \
#   --rounds 30 \
#   --local_learning_rate 0.001 \
#   --local_batch_size 128 \
#   --local_epochs 1 \
#   --default_temp 2.0 \
#   --alpha_dirichlet 100 


# An example command on Spyder Console

# !python main.py \
#   --setup local \
#   --num_clients 1 \
#   --local_model_name CNN \
#   --num_train_samples 3000 \
#   --num_test_samples 500 \
#   --dataset cifar10 \
#   --Foundation_model openai/clip-vit-base-patch32 \
#   --rounds 30 \
#   --local_learning_rate 0.001 \
#   --local_batch_size 128 \
#   --local_epochs 1 \
#   --default_temp 2.0 \
#   --alpha_dirichlet 100











