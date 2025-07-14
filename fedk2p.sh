#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1  # t4 or v100 or a100 or dgx or a5000 or h100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # 8, 16
#SBATCH --mem=16G               # memory per node (ex: 16G) you can get more 
#SBATCH --time=01:00:00 		   # time period you need for your code (it is 12 hours for example)
#SBATCH --mail-user=<jamal73sm@gmail.com> 	# replace with your email address to get emails to know when it is started or failed. 
#SBATCH --mail-type=ALL

#cd /home/shahab33/projects/def-arashmoh/shahab33/FeDK2P
cd /project/def-arashmoh/shahab33/Rohollah/projects/FeD2P


module purge
module load python
module load cuda

source /home/shahab33/fed2p/bin/activate  	# activate your environment
#source /project/def-arashmoh/shahab33/fed2p/bin/activate 

python main.py   	# this is the direction and the name of your code
