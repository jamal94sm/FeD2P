import numpy as np
import transformers
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from Config import get_args
args = get_args()




##############################################################################################################
##############################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict

class clip_plus_linear_head(nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes, device):
        super(clip_plus_linear_head, self).__init__()
        self.FM = FM.to(device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_classes = num_classes
        self.name_classes = name_classes
        self.device = device

        for p in self.FM.parameters():
            p.requires_grad = False

        self.logit_scale = nn.Parameter(torch.tensor(self.FM.config.logit_scale_init_value).to(device))

        embedding_dim = self.FM.get_image_features(torch.randn(1, 3, 224, 224).to(device)).shape[-1]
        self.linear_head = nn.Linear(embedding_dim, num_classes).to(device)

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        image_features = self.FM.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)
        logits = self.linear_head(image_features) * self.logit_scale.exp()
        return logits

    def inference(self, dataset_dict: DatasetDict):
        self.eval()
        logits_list = []

        for sample in dataset_dict["train"]:
            image = sample["image"]
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.FM.get_image_features(**inputs)
                image_features = F.normalize(image_features, dim=-1)
                logits = self.linear_head(image_features) * self.logit_scale.exp()

            logits_list.append(logits.squeeze(0).cpu())  # remove batch dim and move to CPU

        return torch.stack(logits_list)  # shape: [num_samples, num_classes]




##############################################################################################################
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
class LLM(torch.nn.Module):
    def __init__(self):
        super(LLM, self).__init__()
        model_name = "distilgpt2"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    def generate_response(self, max_length=100):
        inputs = self.tokenizer(args.prompt_template, return_tensors="pt").to(args.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
##############################################################################################################
##############################################################################################################

class Image_prompting_plus_Fm(nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes):
        super(Image_prompting_plus_Fm, self).__init__()
        self.FM = FM.to(args.device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_classes = num_classes
        self.name_classes = name_classes

        for p in self.FM.parameters():
            p.requires_grad = False

        self.logit_scale = nn.Parameter(torch.tensor(self.FM.config.logit_scale_init_value))
        self.load_descriptions()
        self.generate_text_rep()
        self.generate_basic_text_rep()

        hidden_size = self.FM.vision_model.config.hidden_size
        self.soft_prompts = nn.Parameter(torch.randn(1, args.num_prompts, hidden_size) * 0.02) # 4= num_prompts
        self.prompt_pos_embed = nn.Parameter(torch.zeros(1, args.num_prompts, hidden_size))
        nn.init.trunc_normal_(self.prompt_pos_embed, std=0.02)

    def load_descriptions(self):
        df = pd.read_csv("Descriptions_Dataset.csv")
        df['descriptions'] = df['descriptions'].str.strip('\'"')
        self.descript_dataset = {
            'descriptions': list(df['descriptions'].values),
            'label': list(df['label'].values)
        }

    @torch.no_grad()
    def generate_text_rep(self):
        if "M" in args.setup:
            class_descriptions = self.descript_dataset['descriptions']
            self.labels = torch.tensor(self.descript_dataset['label'], device=args.device)
        else:
            class_descriptions = [args.prompt_template.format(name) for name in self.name_classes]
            self.labels = torch.arange(self.num_classes, device=args.device)

        tok = self.tokenizer(class_descriptions, padding=True, truncation=True, return_tensors="pt").to(args.device)
        self.text_rep = F.normalize(self.FM.get_text_features(tok["input_ids"]), dim=-1)

    @torch.no_grad()
    def generate_basic_text_rep(self):
        if "mean" in args.setup and "M" in args.setup:
            self.basic_text_rep = torch.stack([
                self.text_rep[(self.labels == n).nonzero(as_tuple=True)[0]].mean(dim=0)
                for n in range(self.num_classes)
            ])
        else:
            class_prompts = [args.prompt_template.format(name) for name in self.name_classes]
            tok = self.tokenizer(class_prompts, padding=True, truncation=True, return_tensors="pt").to(args.device)
            self.basic_text_rep = F.normalize(self.FM.get_text_features(tok["input_ids"]), dim=-1)

    def patchify(self, img, patch_size):
        B, C, H, W = img.shape
        assert H % patch_size == 0 and W % patch_size == 0
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, patches.shape[1], -1)
        return patches

    def _embed_image_with_prompts(self, pixel_values):
        vision = self.FM.vision_model
        device = pixel_values.device
        dtype = pixel_values.dtype
        B = pixel_values.shape[0]

        # 1) Patch embedding -> (B, N, hidden)
        patch_embeds = vision.embeddings.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (B, N, H)

        # 2) Class token -> (B, 1, hidden)
        cls_token = vision.embeddings.class_embedding.to(dtype=dtype)
        cls_tokens = cls_token.expand(B, 1, -1)

        # 3) Visual soft prompts -> (B, P, hidden)
        prompt_tokens = self.soft_prompts.to(device=device, dtype=dtype).expand(B, -1, -1)

        # 4) Positional embeddings (handle both nn.Embedding and Tensor/Parameter)
        pos_module = vision.embeddings.position_embedding
        if isinstance(pos_module, nn.Embedding):
            # Build base position ids for [CLS] + patches
            seq_len_base = 1 + patch_embeds.size(1)
            position_ids = torch.arange(seq_len_base, device=device).unsqueeze(0).expand(B, -1)  # (B, 1+N)
            base_pos = pos_module(position_ids).to(dtype=dtype)  # (B, 1+N, hidden)
        else:
            # pos_module is a Tensor/Parameter with shape (1, 1+N, hidden)
            base_pos = pos_module.to(device=device, dtype=dtype).expand(B, -1, -1)  # (B, 1+N, hidden)

        pos_cls = base_pos[:, :1, :]                 # (B, 1, hidden)
        pos_patches = base_pos[:, 1:, :]             # (B, N, hidden)
        pos_prompts = self.prompt_pos_embed.to(device=device, dtype=dtype).expand(B, -1, -1)  # (B, P, hidden)

        # 5) Compose tokens and add positions: [CLS] + [PROMPTS] + [PATCHES]
        tokens = torch.cat([cls_tokens, prompt_tokens, patch_embeds], dim=1)  # (B, 1+P+N, hidden)
        pos = torch.cat([pos_cls, pos_prompts, pos_patches], dim=1)           # (B, 1+P+N, hidden)
        hidden_states = tokens + pos

        # 6) Vision transformer forward
        hidden_states = vision.pre_layrnorm(hidden_states)
        encoder_out = vision.encoder(hidden_states)[0]        # (B, 1+P+N, hidden)
        pooled = vision.post_layernorm(encoder_out[:, 0, :])  # CLS -> (B, hidden)

        # 7) Project to CLIP space and normalize
        image_embeds = self.FM.visual_projection(pooled)      # (B, proj_dim)
        return F.normalize(image_embeds, dim=-1)


    def forward(self, x, inference=None):
        if isinstance(x, torch.Tensor):
            pixel_values = x.to(args.device)
        else:
            pixel_values = self.processor(images=x, return_tensors="pt")["pixel_values"].to(args.device)

        img_rep = self._embed_image_with_prompts(pixel_values)
        img_rep = img_rep / img_rep.norm(p=2, dim=-1, keepdim=True)

        
        
        #indices = [  np.random.choice((self.labels == n).nonzero(as_tuple=True)[0]) for n in range(self.num_classes)  ] 
        indices = [int((self.labels == n).nonzero(as_tuple=True)[0][torch.randint(0, (self.labels == n).sum(), (1,))]) for n in range(self.num_classes)]
        selected_text_rep = self.text_rep[indices]


        
        if inference:
            text_rep = self.basic_text_rep/self.basic_text_rep.norm(p=2, dim=-1, keepdim=True)
        else: 
            text_rep = selected_text_rep / selected_text_rep.norm(p=2, dim=-1, keepdim=True)
         
            
         
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_rep @ text_rep.t()
        return logits
        





##############################################################################################################
##############################################################################################################

class Prompt_Generator_plus_FM(torch.nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes):
        super(Prompt_Generator_plus_FM, self).__init__()
    
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
        
        
        if args.generator_name == "CNN-Transpose":
            self.pgen = Prompt_Generator_CNNT(input_size = self.FM.config.projection_dim, output_size = self.FM.config.vision_config.image_size).to(args.device)
        elif args.generator_name == "MLP":
            self.pgen = Prompt_Generator_MLP(input_size = self.FM.config.projection_dim, output_size = self.FM.vision_model.config.hidden_size).to(args.device)
        elif args.generator_name=="AttentionModel":
            self.pgen = Prompt_Generator_AttentionModel(512, 8, self.text_rep)
        else:
            raise ValueError("This is a custom error message.")

      
            
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
        



    def __call__(self, x, inference=False):

        out = self.pgen(x)

        if args.generator_name == "CNN-Transpose":
            img_rep = self.FM.get_image_features(out)

        if args.generator_name == "MLP" or args.generator_name == "AttentionModel":
            pooled_output = self.FM.vision_model.encoder(out.unsqueeze(dim=1), return_dict=False)[0][:, 0, :]
            pooled_output = self.FM.vision_model.post_layernorm(pooled_output)
            img_rep = self.FM.visual_projection(pooled_output)

        indices = [  np.random.choice((self.labels == n).nonzero(as_tuple=True)[0]) for n in range(self.num_classes)] 
        selected_text_rep = self.text_rep[indices]

        img_rep = img_rep / img_rep.norm(p=2, dim=-1, keepdim=True)
        
        if inference:
            text_rep = self.basic_text_rep/self.basic_text_rep.norm(p=2, dim=-1, keepdim=True)
        else: 
            text_rep = selected_text_rep / selected_text_rep.norm(p=2, dim=-1, keepdim=True)
         
            
         
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_rep @ text_rep.t()
        return logits
##############################################################################################################
##############################################################################################################
class Prompt_Generator_CNNT(nn.Module):
    def __init__(self, input_size = 512, output_size = 224):
        super(Prompt_Generator_CNNT, self).__init__()
        self.fc1 = nn.Linear(input_size, 32*7*7)
        # Define transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=8, stride=4, padding=2)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)  
        self.deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)  
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 32, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x) 
        #x = torch.sigmoid(x)
        return x
##############################################################################################################
##############################################################################################################   
class Prompt_Generator_MLP(nn.Module):
    def __init__(self, input_size=512, output_size=768):
        super(Prompt_Generator_MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)  # First layer
        self.layer2 = nn.Linear(512, 256)         # Second layer
        self.layer3 = nn.Linear(256, 128)          # Third layer
        self.layer4 = nn.Linear(128, output_size)  # Fourth layer
        self.relu = nn.ReLU()                     # ReLU activation function
        self.dropout = nn.Dropout(0.5)            # Dropout for regularization
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.layer4(x)  # No activation on the output layer
        return x
##############################################################################################################
##############################################################################################################
class Prompt_Generator_AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, keys):
        super(Prompt_Generator_AttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dense = nn.Linear(embed_dim, 768)
        self.base_keys = keys
        self.keys = keys.repeat(10, 1, 1)
    def forward(self, queries):
        queries = queries.unsqueeze(1)
        if queries.shape[0] != self.keys.shape[0]: 
            self.keys = self.base_keys.repeat(queries.shape[0], 1, 1)
        attn_output, _ = self.attention(queries, self.keys, self.keys)
        y = self.dense(attn_output)
        return y.squeeze(1)
##############################################################################################################
##############################################################################################################
class VGGBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        return x

##############################################################################################################
##############################################################################################################
class LightWeight_CNN(nn.Module):
    def __init__(self, input_shape, output_shape, num_vcg):
        super().__init__()
        self.num_vcg = num_vcg
        self.vgg_block1 = VGGBlock2(input_shape[0], 32)
        self.vgg_block2 = VGGBlock2(32, 64)
        self.vgg_block3 = VGGBlock2(64, 64)
        if self.num_vcg==1: self.fc1 = nn.Linear(int(input_shape[1]*input_shape[2]*32/4), 512)
        elif self.num_vcg==2: self.fc1 = nn.Linear(int(input_shape[1]*input_shape[2]*64/16), 512)
        elif self.num_vcg==3: self.fc1 = nn.Linear(int(input_shape[1]*input_shape[2]*64/64), 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        if self.num_vcg>=1:
            x = self.vgg_block1(x)
            if self.num_vcg>=2:
                x = self.vgg_block2(x)
                if self.num_vcg>=3:
                    x = self.vgg_block3(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


##############################################################################################################
##############################################################################################################
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels) )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

##############################################################################################################
##############################################################################################################
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
##############################################################################################################
##############################################################################################################

from torchvision.models import mobilenet_v2

def MobileNetV2(input_shape, output_shape):
    # Load the base MobileNetV2 model
    model = mobilenet_v2(pretrained=False)

    # Adjust the first convolution layer if input channels differ
    in_channels = input_shape[0]
    if in_channels != 3:
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # Adjust the classifier for the desired output shape
    model.classifier[1] = nn.Linear(model.last_channel, output_shape)

    return model
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


##############################################################################################################
##############################################################################################################
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

class ResNet10(ResNet):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10):
        # ResNet10 has [1, 1, 1, 1] blocks in each layer
        super(ResNet10, self).__init__(block=BasicBlock, layers=[1, 1, 1, 1])
        
        # Adjust for small input sizes
        if input_shape[1] < 64 or input_shape[2] < 64:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
##############################################################################################################
##############################################################################################################
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

class ResNet20(ResNet):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(ResNet20, self).__init__(block=BasicBlock, layers=[3, 3, 3, 0])

        if input_shape[1] < 64 or input_shape[2] < 64:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


##############################################################################################################
##############################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(EfficientNet, self).__init__()

        # Load pretrained EfficientNet-B0 backbone
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()  # Remove original classifier

        # Determine the number of features output by the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_input_resized = nn.functional.interpolate(dummy_input, size=(224, 224), mode='bilinear')
            features = self.backbone(dummy_input_resized)
            feature_dim = features.shape[1]

        # Custom classifier
        self.fc1 = nn.Linear(feature_dim, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Resize input to match EfficientNet expected input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

##############################################################################################################
##############################################################################################################
