import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import numpy as np
import copy
from torchvision import datasets, models, transforms
from dataloader import load_data
#import matplotlib.pyplot as plt


class VGG(object):

    def __init__(self, pretrained_model, num_classes=6, learning_rate=0.0001, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        """ Initialize a new network based on VGG architecture"""
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.model = pretrained_model
        self.num_classes = num_classes
        self.lr = learning_rate
        self.weight_scale = weight_scale

        # Freeze all layers 
        for param in self.model.features.parameters():
            param.require_grad = False
        
        # Newly created modules have require_grad=True by default
        num_features = self.model.classifier[6].in_features
        features = list(self.model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, num_classes)]) # Add final linear layer

        self.model.classifier = nn.Sequential(*features) # Replace the model classifier
        #print(self.model)
        
    def loss(self, X, y=None):
        """ Evaluate loss and gradient for the network """
        pass


    def train(self, dataloaders, dataset_sizes, num_epochs = 25):

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        #train_batches = len(dataloaders["train"])
        #val_batches = len(dataloaders["val"])
    
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

        for epoch in range(0, num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs-1))
            print('-'*10)
            
            for mode in ['train', 'val']:
                # Set model to  the appropriate mode
                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval() 
                    
                total_loss = 0.0
                total_correct = 0 

                for inputs, labels in dataloaders[mode]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Zero out the gradient for this next pass
                    optimizer.zero_grad()

                    # Forward pass
                    # Track history if in train
                    with torch.set_grad_enabled(mode == 'train'):
                        outputs = self.model(inputs)
                        _, y_preds = torch.max(outputs, 1)

                        # Compute loss
                        loss = loss_fn(outputs, labels)
                
                        if mode == "train":
                            # Compute gradient and take gradient descent step
                            loss.backward() 
                            optimizer.step()
                
                    # statistics
                    total_loss += loss.item() * inputs.size(0)
                    total_correct += torch.sum(y_preds == labels.data)
                
            epoch_loss = total_loss / dataset_sizes[mode]
            epoch_acc = total_correct.double() / dataset_sizes[mode]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))
            
            # deep copy the model
            if mode == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model



if __name__ == "__main__":
    
    # Load the data
    pathname = "/Users/sarahciresi/Documents/GitHub/CS231n-Project-2019/datasets/trashnet/data"
    dataloaders, dataset_sizes, class_names = load_data(pathname)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg16 = models.vgg16(pretrained=True).to(device)

    # Initialize the model
    model = VGG(vgg16)

    # Train the last layer of the model
    model = model.train(dataloaders, dataset_sizes, num_epochs=25)

    


    

'''
### models ###
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenetv2(pretrained=True)
####
'''
