import MyModels
import MyUtils
import MyDatasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from Config import args
from torch.utils.data import DataLoader, TensorDataset



##############################################################################################################
##############################################################################################################
class Server():
    def __init__(self, model, clients):
        self.model = model
        self.clients = clients
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.local_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.Loss = []
        
        

    
    def get_generated_images(self):
        with torch.no_grad():
            out = self.model.pgen(self.model.z) 
        return out
    
    def train_decoder(self, num_classes):
        self.zs = []
                
        for c in range(num_classes):            
            print(30*"-", f"Training for class {c}", 30*"-")

            self.model.z = torch.nn.Parameter(torch.randn(args.num_generated_images, 256, 16, 16, device=args.device))   
            
            label = c*torch.ones( args.num_generated_images).to(args.device)


            for step in range(args.global_epochs):
                self.optimizer.zero_grad()
                
                logits = self.model() 
                loss = self.loss_fn(logits, label.long())
                
                loss.backward()
                self.optimizer.step()
            
            
            self.zs.append(self.model.z)
            

    
    def distill_generator(self, logits):
        teacher_knowledge = logits
        
        if self.model.labels.shape[0] != teacher_knowledge.shape[0]:
            data_for_extension = { "train": {"image":self.model.text_rep, "label":self.model.labels} }
            teacher_knowledge = MyUtils.extend_proto_outputs_to_labels(data_for_extension, teacher_knowledge)
        
        extended_data = MyDatasets.ddf({
            "student_model_input": self.model.text_rep,
            "student_model_output": self.model.labels,
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





##############################################################################################################
##############################################################################################################
class Device():
    def __init__(self, ID, data, num_classes, name_classes):
        self.ID = ID
        self.data = data
        self.num_classes = num_classes
        self.name_classes = name_classes
        self.num_samples = torch.bincount(self.data["train"]["label"], minlength=num_classes).to(args.device)


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




        MyUtils.Model_Size(self.model)


        if args.load_saved_models: 
            self.model.load_state_dict(torch.load("/content/drive/MyDrive/FedD2P/Trained_Models/ResNet.zip", weights_only=True))
            print("model {} loaded succesfully for client {}".format(args.local_model_name, self.ID))
            #torch.save(clients[0].model.state_dict(), '/content/drive/MyDrive/FedD2P/Trained_Models/ResNet')
            

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.local_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.Loss = []
        self.Acc = []
        self.test_Acc = []
    def local_training(self):
        a,b, c = MyUtils.Train(self.model, self.data, self.optimizer, self.scheduler, self.loss_fn,
                               args.local_batch_size, args.local_epochs, args.device, args.debug)
        self.Loss += a
        self.Acc += b
        self.test_Acc += c
        print(self.ID, b[-1], c[-1])
    def local_distillation(self, teacher_knowledge, prototype=True):
        if prototype:
            teacher_knowledge = MyUtils.extend_proto_outputs_to_labels(self.data, teacher_knowledge)
        extended_data = MyDatasets.ddf({"student_model_input": self.data["train"]["image"], 
                                        "student_model_output":self.data["train"]["label"], 
                                        "teacher_knowledge": teacher_knowledge}
                                      )
        a, b, c = MyUtils.Distil(self.model, extended_data, self.data, self.optimizer, self.scheduler, self.loss_fn,
                                 args.local_batch_size, args.local_epochs, args.device, args.debug)
        self.Loss += a
        self.Acc += b
        self.test_Acc += c
        print("====>",self.ID, b[-1], c[-1])
    def cal_proto_logits_orginal(self, batch_size=32):
        logits = self.model(self.data["train"]["image"].to(args.device))
        labels = self.data["train"]["label"].to(args.device)

  


        if "sift" in args.setup:
            print(30*'here')
            predicted = torch.argmax(logits, dim=1)
            correct_mask = (predicted == labels)
            missing_classes = torch.tensor([cls.item() for cls in labels.unique() if cls not in labels[correct_mask].unique()]).to(args.device)
            missing_class_mask = torch.isin(labels, missing_classes)
            final_mask = correct_mask | missing_class_mask
            logits = logits[final_mask]
            labels = labels[final_mask]



        unique_classes = sorted(set(labels.tolist()))        
        num_classes = len(unique_classes)

        self.proto_logits = torch.empty((num_classes, num_classes), device=logits.device)
        
        for c in unique_classes:
            mask = labels  == c
            category_logits = logits[mask].mean(dim=0)
            self.proto_logits[c] = category_logits

    def cal_proto_logits(self, batch_size=64):
        images = self.data["train"]["image"]
        labels = self.data["train"]["label"]

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size)

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

        if "sift" in args.setup:
            predicted = torch.argmax(logits, dim=1)
            correct_mask = (predicted == labels)
            missing_classes = torch.tensor(
                [cls.item() for cls in labels.unique() if cls not in labels[correct_mask].unique()]
            ).to(args.device)
            missing_class_mask = torch.isin(labels, missing_classes)
            final_mask = correct_mask | missing_class_mask
            logits = logits[final_mask]
            labels = labels[final_mask]



        self.proto_logits = torch.empty((num_classes, num_classes), device=logits.device)

        
        for c in unique_classes:
            mask = labels == c
            category_logits = logits[mask].mean(dim=0)
            self.proto_logits[c] = category_logits


##############################################################################################################
##############################################################################################################



