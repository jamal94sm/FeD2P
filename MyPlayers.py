import MyModels
import MyUtils
import MyDatasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from Config import args
from torch.utils.data import DataLoader, TensorDataset
from datasets import DatasetDict, concatenate_datasets


##############################################################################################################
##############################################################################################################
class Server():
    def __init__(self, model, clients, public_data):
        self.model = model
        self.clients = clients
        self.public_data = public_data
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.local_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.Loss = []
        
    def aggregation(self):
        coeficients = 1 / torch.stack([client.num_samples for client in self.clients]).sum(dim=0) 
        summ = torch.stack([client.logits * client.num_samples for client in self.clients]).sum(dim=0)
        self.ave_logits = summ * coeficients 
        return self.ave_logits

    def fedavg_aggregation_and_implanting(self):
        Models = [client.model for client in self.clients]
        global_dict = Models[0].state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
        for model in Models:
            local_dict = model.state_dict()
            for key in global_dict:
                global_dict[key] += local_dict[key]
        for key in global_dict:
            global_dict[key] = global_dict[key] / len(Models)
        for client in self.clients:
            client.model.load_state_dict(global_dict)

    def get_general_knowledge(self):
        with torch.no_grad():
            pred = self.model(self.public_data["train"]["image"].to(args.device), inference=True)
        return pred

    def just_training(self):
        
        data = MyDatasets.ddf({"train": {
            "image": self.model.text_rep,
            "label": self.model.labels,
        }})
        
        loss, _, _ = MyUtils.Just_Train(
            model = self.model,
            data = data,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_fn = self.loss_fn,
            batch_size = args.global_batch_size if "M" in args.setup else 8,
            epochs = args.global_epochs,
            device = args.device,
            debug = args.debug
        )
        self.Loss += loss

    
    def distill_generator(self, data, logits):
        teacher_knowledge = logits
        
        data_for_extension = { "train": {"image": data["train"]["image"], "label": data["train"]["label"] } }
        teacher_knowledge = MyUtils.extend_proto_outputs_to_labels(data_for_extension, teacher_knowledge)
        
        extended_data = MyDatasets.ddf({
            "student_model_input":  data["train"]["image"],
            "student_model_output": data["train"]["label"],
            "teacher_knowledge": teacher_knowledge
        })

        loss, _, _ = MyUtils.Distil(
            model = self.model,
            extended_data = extended_data,
            data = None,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_fn = self.loss_fn,
            batch_size = args.global_batch_size if "M" in args.setup else 8,
            epochs = args.global_epochs,
            device = args.device,
            debug = args.debug
        )
        self.Loss += loss

    
    def fedavg_aggregation(self):
        import copy
        import torch

        Models = [client.model for client in self.clients]
        global_dict = copy.deepcopy(Models[0].state_dict())

        # Initialize global_dict to zeros
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])

        # Sum all local model parameters
        for model in Models:
            local_dict = model.state_dict()
            for key in global_dict:
                global_dict[key] += local_dict[key]

        # Average the parameters
        for key in global_dict:
            global_dict[key] = global_dict[key] / len(Models)

        # Create a new model and load the averaged state_dict
        global_model = copy.deepcopy(Models[0])  # assumes all models share the same architecture
        global_model.load_state_dict(global_dict)

        return global_model

    
    def zero_shot(self, data, FM, processor, tokenizer, proto=False, batch_size=16):
        
        processor.image_processor.do_rescale = False
        processor.image_processor.do_normalize = False

        device = args.device
        images = data["image"]
        labels = data["label"]
        
        
        img_reps = []

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            inputs = processor(
                text=["blank"] * len(batch_images),
                images=batch_images,
                return_tensors="pt"
                )

            with torch.no_grad():
                pixel_values = inputs["pixel_values"].to(device)
                batch_img_rep = FM.get_image_features(pixel_values)
                batch_img_rep = batch_img_rep / batch_img_rep.norm(p=2, dim=-1, keepdim=True)
                img_reps.append(batch_img_rep.cpu())  # Move back to CPU to save GPU memory

            del inputs, pixel_values, batch_img_rep
            torch.cuda.empty_cache()

        img_rep = torch.cat(img_reps, dim=0).to(device)

        with torch.no_grad():
            text_rep = self.model.basic_text_rep / self.model.basic_text_rep.norm(p=2, dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * img_rep @ text_rep.t()

        if not proto:
            return logits

        unique_classes = sorted(set(labels.tolist()))
        num_classes = len(unique_classes)
        proto_logits = torch.empty((num_classes, num_classes), device=logits.device)

        for c in unique_classes:
            mask = (labels == c)
            category_logits = logits[mask].mean(dim=0)
            proto_logits[c] = category_logits

        return proto_logits


##############################################################################################################
##############################################################################################################
class Device():
    def __init__(self, ID, data, num_classes, name_classes, public_data):
        self.ID = ID
        self.data = data

        self.num_classes = num_classes
        self.name_classes = name_classes
        self.num_samples = torch.bincount(self.data["train"]["label"], minlength=num_classes).to(args.device)
        self.public_data = public_data

        
        if args.local_model_name=="MLP": #MLP
            self.model = MyModels.MLP(data["train"]["image"].view(data["train"]["image"].size(0), -1).size(1), self.num_classes).to(args.device)
        elif args.local_model_name=="ResNet": 
            self.model = MyModels.ResNet([1, 1, 1], self.num_classes).to(args.device) #ResNet
        elif args.local_model_name=="CNN": 
            self.model = MyModels.LightWeight_CNN(data["train"]["image"][0].shape, self.num_classes, 3).to(args.device) #CNN
        elif args.local_model_name=="MobileNetV2":
            self.model = MyModels.MobileNetV2(data["train"]["image"][0].shape, self.num_classes).to(args.device) #MobileNetV2
        elif args.local_model_name=="ResNet18":
            self.model = MyModels.ResNet18(data["train"]["image"][0].shape, self.num_classes).to(args.device) #ResNet18
        elif args.local_model_name=="ResNet10":
            self.model = MyModels.ResNet10(data["train"]["image"][0].shape, self.num_classes).to(args.device) #ResNet10
        elif args.local_model_name=="ResNet20":
            self.model = MyModels.ResNet20(data["train"]["image"][0].shape, self.num_classes).to(args.device) #ResNet20
        elif args.local_model_name=="EfficientNet":
            self.model = MyModels.EfficientNet(data["train"]["image"][0].shape, self.num_classes).to(args.device) #EfficientNet

        MyUtils.model_size(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.local_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.Loss = []
        self.Acc = []
        self.test_Acc = []

    def local_training(self):
        a,b, c = MyUtils.Train(self.model, self.data, self.optimizer, self.scheduler, self.loss_fn,
                               args.local_batch_size, args.local_epochs, args.device, args.debug, eval=True)
        self.Loss += a
        self.Acc += b
        self.test_Acc += c


    def local_distillation(self, data, teacher_knowledge, proto=False):
        if proto:
            teacher_knowledge = MyUtils.extend_proto_outputs_to_labels(data, teacher_knowledge)

        min_len = min(
            len(data["train"]["image"]),
            len(data["train"]["label"]),
            len(teacher_knowledge)
        )

        extended_data = MyDatasets.ddf({
            "student_model_input": data["train"]["image"][:min_len],
            "student_model_output": data["train"]["label"][:min_len],
            "teacher_knowledge": teacher_knowledge[:min_len]
        })
        '''
        extended_data = MyDatasets.ddf({"student_model_input": data["train"]["image"], 
                                        "student_model_output": data["train"]["label"], 
                                        "teacher_knowledge": teacher_knowledge}
                                      )
        '''
        a, b, c = MyUtils.Distil(self.model, extended_data, self.data, self.optimizer, self.scheduler, self.loss_fn,
                                 args.local_batch_size, args.local_epochs, args.device, args.debug)
        


    def cal_logits(self, data, proto=False, sifting=False):
        images = data["train"]["image"]
        labels = data["train"]["label"]

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=64)

        all_logits = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(args.device)
                batch_labels = batch_labels.to(args.device)
                logits = self.model(batch_images)
                all_logits.append(logits)
                all_labels.append(batch_labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        unique_classes = sorted(set(labels.tolist()))
        num_classes = len(unique_classes)

        if sifting:
            predicted = torch.argmax(logits, dim=1)
            correct_mask = (predicted == labels)
            missing_classes = torch.tensor(
                [cls.item() for cls in labels.unique() if cls not in labels[correct_mask].unique()]
            ).to(args.device)
            missing_class_mask = torch.isin(labels, missing_classes)
            final_mask = correct_mask | missing_class_mask
            logits = logits[final_mask]
            labels = labels[final_mask]

        if not proto: 
            self.logits = logits
        else:
            self.logits = torch.empty((num_classes, num_classes), device=logits.device)
            for c in unique_classes:
                mask = labels == c
                category_logits = logits[mask].mean(dim=0)
                self.logits[c] = category_logits
        
    def local_selective_training(self, data, eval=True):
        #merged = DatasetDict({
        #    "train": concatenate_datasets([self.data["train"], self.public_data["train"]]),
        #    "test": self.data["test"]  # Only ds1 has a test split
        #})

        a,b, c = MyUtils.Train(self.model, data, self.optimizer, self.scheduler, self.loss_fn, args.local_batch_size, args.local_epochs, args.device, args.debug, eval=eval)
        self.Loss += a
        self.Acc += b
        self.test_Acc += c



##############################################################################################################
##############################################################################################################



