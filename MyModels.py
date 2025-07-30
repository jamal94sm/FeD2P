import numpy as np
import transformers
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import args
import pandas as pd
import matplotlib.pyplot as plt





##############################################################################################################
def load_clip_model():
    model_name = args.Foundation_model
    model = transformers.CLIPModel.from_pretrained(model_name)
    processor = transformers.CLIPProcessor.from_pretrained(model_name, use_fast=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if "BN" in args.setup: 
        print("Unfreeze LayerNorm layers in the image encoder")

        # Unfreeze LayerNorm layers in the image encoder
        for module in model.vision_model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.train()  # Set to training mode
                for param in module.parameters():
                    param.requires_grad = True
                    
    return model, processor, tokenizer

##############################################################################################################
##############################################################################################################
class Decoder_plus_FM(torch.nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes):
        super(Decoder_plus_FM, self).__init__()
    
        self.FM = FM.to(args.device)
        self.tokenizer = tokenizer
        self.processor = processor
        for param in self.FM.parameters(): param.requires_grad = False
        self.num_classes = num_classes
        self.name_classes = name_classes
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.FM.config.logit_scale_init_value))
        self.load_descriptins()
        self.generate_text_rep()
        self.generate_basics()
        
            
        

        if args.generator_name== "VAEDecoder":
            self.pgen = VAEDecoder().to(args.device)
            for param in self.pgen.parameters(): param.requires_grad = False
            self.z = nn.Parameter(torch.randn(64, 4, 28, 28, device=args.device))
        elif args.generator_name== "VQGAN":
            self.pgen = VQGANDecoder().to(args.device)
            for param in self.pgen.parameters(): param.requires_grad = False
            #self.z = nn.Parameter(torch.randn(args.num_generated_images, 256, 16, 16, device=args.device))   
            
            
    def load_descriptins(self):
        df = pd.read_csv("Descriptions_Dataset.csv")
        df['descriptions'] = df['descriptions'].str.strip('\'"')
        self.descript_dataset = {
            'descriptions': list(df['descriptions'].values),
            'label': list(df['label'].values)
        }
    
        
    def generate_text_rep(self):
        if "M" in args.setup:
            class_descriptions = self.descript_dataset['descriptions']
            self.labels = torch.tensor(self.descript_dataset['label'])
        else:
            class_descriptions = [args.prompt_template.format(name) for name in self.name_classes]
            self.labels = torch.arange(self.num_classes)

        input_ids = self.tokenizer( class_descriptions,  add_special_tokens=True, padding=True,  truncation=True,  return_tensors="pt" ).to(args.device)        
        with torch.no_grad():
            self.text_rep = self.FM.get_text_features(input_ids["input_ids"])


    def generate_basics(self):
        if "mean" in args.setup:
            with torch.no_grad():
                self.basic_text_rep = torch.stack([self.text_rep[(self.labels == n).nonzero(as_tuple=True)[0]].mean(dim=0) for n in range(self.num_classes)])

        else:
            class_descriptions = [args.prompt_template.format(name) for name in self.name_classes]
            input_ids = self.tokenizer( class_descriptions,  add_special_tokens=True, padding=True,  truncation=True,  return_tensors="pt" ).to(args.device)        
            with torch.no_grad():
                self.basic_text_rep = self.FM.get_text_features(input_ids["input_ids"])       
        



    def __call__(self, inference=False):
        
        out = self.pgen(self.z)

    



        img_rep = self.FM.get_image_features(out)
        img_rep = img_rep / img_rep.norm(p=2, dim=-1, keepdim=True)
        
        
        if inference:
            text_rep = self.basic_text_rep/self.basic_text_rep.norm(p=2, dim=-1, keepdim=True)
        else: 
            indices = [  np.random.choice((self.labels == n).nonzero(as_tuple=True)[0]) for n in range(self.num_classes)] 
            selected_text_rep = self.text_rep[indices]
            text_rep = selected_text_rep / selected_text_rep.norm(p=2, dim=-1, keepdim=True)
            
            
            
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_rep @ text_rep.t()
        return logits


##############################################################################################################
##############################################################################################################   


import torch
from torch import nn
from diffusers import AutoencoderKL

class VAEDecoder(nn.Module):
    def __init__(self, model_name="stabilityai/sd-vae-ft-mse"):
        super(VAEDecoder, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.vae.eval()  # Set to evaluation mode

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            decoded = self.vae.decode(latents).sample
        return decoded


##############################################################################################################
##############################################################################################################   
import torch
import torch.nn as nn
import os
import sys
sys.path.append("~/taming-transformers/")
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf

class VQGANDecoder(nn.Module):
    def __init__(self, config_path="taming-transformers/logs/vqgan_imagenet_f16_16384/configs/model.yaml", ckpt_path="taming-transformers/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt", device=None):
        super(VQGANDecoder, self).__init__()
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load config and checkpoint
        config = OmegaConf.load(config_path)
        params = config.model.params

        # Initialize model
        self.model = VQModel(**params)
        with torch.serialization.safe_globals([]):  # or weights_only=False if trusted
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict["state_dict"], strict=False)
        self.model.eval().to(self.device)

    def forward(self, z):

        with torch.no_grad():
            x_rec = self.model.decode(z)
            x_rec = torch.clamp((x_rec + 1.0) / 2.0, 0.0, 1.0)
            x_rec = F.interpolate(x_rec, size=(224, 224), mode='bilinear', align_corners=False)
        return x_rec






##############################################################################################################
##############################################################################################################
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights


def ResNet18(input_shape=(3, 224, 224), num_classes=10, pretrained=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Adjust for small input sizes
    if input_shape[1] < 64 or input_shape[2] < 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    return model

