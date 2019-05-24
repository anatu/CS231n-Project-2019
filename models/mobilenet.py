import numpy as np
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, models, transforms
from mobilenet_basemodel import MobileNetV2
import sys
sys.path.insert(0, '../utils')
from dataloader import load_data
from viz import show_confusion_mat, imshow, create_grid_for_mb
import matplotlib.pyplot as plt

class MobileNet(object):

    def __init__(self, pretrained_path, device, num_classes=7, lr=0.0001, reg=0.0, dtype=np.float32):
        '''
        Initialize a new network based on the MobileNet architecture.
        Uses the MobileNetV2 implementation adapted for PyTorch.
        '''
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_classes = num_classes
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        self.save_model_path = "./model_weights/best_model_weights_mb.pt"
        
        # Load the pretrained model 
        state_dict = torch.load(pretrained_path)
        self.model = MobileNetV2(n_class=1000, device=self.device)
        self.model.load_state_dict(state_dict)

        # Freeze all original layers
        for param in self.model.features.parameters():
            param.require_grad = False

        # Remove last fully connected layer and replace with layer with 7 output classes
        num_features = self.model.classifier[1].in_features
        features = list(self.model.classifier.children())[:-1]                  # Remove last layer
        features.extend([nn.Linear(num_features, num_classes).to(self.device)]) # Add final linear layer
        self.model.classifier = nn.Sequential(*features)                  # Replace the model classifier
        
        
                                                                 
    def train(self, dataloaders, dataset_sizes, num_epochs = 25):
        ''' Function to train the model. Takes as input:                                                         
        - dataloaders:     dataloaders for the train and validation datasets                                       
        - dataset_sizes:   sizes of the train and validation datasets                        
        - num_epochs:      number of epochs to train for                                                          
        '''
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        for epoch in range(0, num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs-1))
            print('-'*10)
            
            for mode in ['train', 'val']:
                # Set model to  the appropriate mode
                if mode == "train":
                    #scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                    
                total_loss = 0.0
                total_correct = 0
                
                for inputs, labels in dataloaders[mode]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero out the gradient for this next pass
                    optimizer.zero_grad()
                    
                    # Forward pass
                    # Track history if in train
                    with torch.set_grad_enabled(mode == 'train'):
                        outputs = self.model(inputs)
                        _, y_preds = torch.max(outputs, 1)
                        
                        # Compute loss
                        loss = self.loss_fn(outputs, labels)
                        
                        if mode == "train":
                            # Compute gradient and take gradient descent step
                            loss.backward()
                            optimizer.step()

                    # Compute statistics
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

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
            
        # Save best model weights to disk for later loading
        torch.save(self.model.state_dict(), self.save_model_path)
            
        return self.model


    def eval_model(self, dataloaders, mode = 'val'):
        ''' Function that evaluates the model on a specified dataset. Takes in                                       
        dictionary of dataloaders and a mode, equal to 'val' or 'test', specifying                                 
        which dataset to evaluate the trained model on.                                                               
        '''
    
        since = time.time()
        avg_loss, avg_acc, total_loss, total_correct = 0,0,0,0
        num_batches = len(dataloaders[mode])
        mode_str = "Validation" if mode == 'val' else "Test"
        
        print("Evaluating model on {} set".format(mode_str))
        print('-' * 10)

        with torch.no_grad():
            for i, data in enumerate(dataloaders[mode]):
                if i % 100 == 0:
                    print("\r{} batch {}/{}".format(mode_str, i, num_batches), end='', flush=True)
                    
                self.model.train(False)
                self.model.eval()
                    
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            
                outputs = self.model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = self.loss_fn(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                total_correct += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
                
        avg_loss = total_loss / dataset_sizes[mode]
        avg_acc = total_correct.double() / dataset_sizes[mode]
        
        elapsed_time = time.time() - since
        print()
        print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Average {} loss     : {:.4f}".format(mode_str, avg_loss))
        print("Average {} accuracy : {:.4f}".format(mode_str, avg_acc))
        print('-' * 10)


    def load_model(self, path, train_mode = False):
        ''' Function to load the model weights from specified path.                                          
        If train_mode is set to true, model will be trained after loading the weights.                             
        Otherwise, model will just be used for evaluation, and model must be set to eval mode.                      
        '''
        
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        
        # Remember to call self.model.eval() if only evaluating with this reloaded model
        if train_mode == False:
            self.model.eval()
            
        return self.model


    def visualize_model(self, num_images=16):
        ''' Function to visualize the model predictions.                                     
        Takes in the parameter num_images, the number of images to display                 
        in each minibatch. num_images should be a square number                  
        for easy plotting of an N x N grid of images.                                                                                    
        Also computes and plots the confusion matrix.                                                                                
        Saves the visualizations to a sibling folder.                                                                                 
        '''
        
        # Set model for evaluation
        self.model.train(False)
        self.model.eval()
        
        images_so_far = 0
        file_path_base = "./mobilenet_visuals/"
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        
        with torch.no_grad():
            for i, data in enumerate(dataloaders['val']):
                inputs, labels = data
                size = inputs.size()[0]
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                # Create and save grid of prediction images for each minibatch
                create_grid_for_mb(i, inputs, num_images, class_names, preds, labels, file_path_base)
                
        # Plot the confusion matrix
        show_confusion_mat(confusion_matrix, self.num_classes, class_names, outfile=file_path_base + "confusion_mat.png")
            

    
if __name__ == "__main__":
    
    # Load the data
    pathname = "/home/sarahciresi/gcloud/project/CS231n-Project-2019/datasets/trashnet/data/dataset-split"
    
    dataloaders, dataset_sizes, class_names = load_data(pathname)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model using weights from pretrained network
    mobilenet_model = MobileNet('mobilenet_v2.pth.tar',  device, num_classes=7)

    # Train the last layer of the model
    mobilenet_model.train(dataloaders, dataset_sizes, num_epochs=5)
    
    # Or load from saved weights from previously retrained model
    ## mobilenet_model.load_model("./best_model_weights.pt_mb", train_mode = False)

    # Some visualizations
    mobilenet_model.visualize_model(num_images=25)

    # Can use eval_model method to evaluate the model after training
    ## mobilenet_model.eval_model(dataloaders, mode = 'val')

                        
