import torch
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf

# Load the YAML config and checkpoint
config_path = "/project/def-arashmoh/shahab33/FeD2P/taming-transformers/logs/vqgan_imagenet_f16_16384/configs/model.yaml"
ckpt_path = "/project/def-arashmoh/shahab33/FeD2P/taming-transformers/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt"

# Load configuration
config = OmegaConf.load(config_path)
model = VQModel(config.model.params)

# Load pretrained weights
state_dict = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(state_dict["state_dict"], strict=False)
model.eval().cuda()  # move model to GPU

# Create a trainable latent z (shape: B x 256 x 16 x 16)
z = torch.randn(1, 256, 16, 16, requires_grad=True).cuda()

# Decode z to synthetic image
with torch.no_grad():
    x_rec = model.decode(z)  # output shape: [1, 3, 256, 256]
    x_rec = torch.clamp((x_rec + 1.0) / 2.0, 0.0, 1.0)  # map to [0, 1] for visualization


