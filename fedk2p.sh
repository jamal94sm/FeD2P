#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1       # Narval: a100, a100_4g.20g; Cedar: p100, p100l, v100l, a40
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1            # adjust (e.g., 8 or 16 if needed)
#SBATCH --mem=40G                    # memory per node
#SBATCH --time=14:00:00               # job time limit (HH:MM:SS)
#SBATCH --mail-user=jamal73sm@gmail.com
#SBATCH --mail-type=ALL

cd /project/def-arashmoh/shahab33/FeD2P #Narval 

module purge
module load python
module load cuda

source /home/shahab33/fed2p/bin/activate #Narval

#python main.py --local_model_name "ResNet18" --dataset "fashion_mnist" --num_train_samples 10000 --alpha_dirichlet 10 --rounds 30 --num_synth_img_per_class 100 --output_name "_ResNet18_Fashion_10K_alpha10_synth100_"
python main.py 
