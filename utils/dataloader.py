# CODE REFERENCES
# https://towardsdatascience.com/how-to-build-an-image-classifier-for-waste-sorting-6d11d3c9c478
# https://github.com/danny95333/Trash-Classification-based-on-CNN
#from linear_classifier import *
#from linear_svm import *
#from softmax import *
import time
import random
import shutil
import re
import os
import torch
from torchvision import transforms, datasets
from pathlib import Path
userPath = os.path.expanduser("~")


######## #### #### #### #### #### #### #### #### #### #### #### ##### #######
###       File that handles all data processing / loading                 ###
### -- If the file itself is run as the main method, it takes resized     ###  
###    images and port into train, val, test folders.                     ###
###                                                                       ###
### -- Otherwise, assuming the above has already been done, the           ###
###    load_data() function in this file can be used to load data         ###
###     of the appropriate split into dataloader objects                  ###
######## #### #### #### #### #### #### #### #### #### #### #### #### ########


def split_indices(folder, seed1, seed2):
    '''
    Function to split indices for a folder into train, validation, and test indices with random sampling.
    Takes as input:
    - folder: folder path
    - seed1, seed2: seeds for random init
    Returns  (train, valid, test) indices
    '''
    n = len(os.listdir(folder))
    full_set = list(range(1,n+1))

    ## train indices
    random.seed(seed1)
    train = random.sample(list(range(1,n+1)),int(.5*n))

    ## temp
    remain = list(set(full_set)-set(train))

    ## separate remaining into validation and test
    random.seed(seed2)
    valid = random.sample(remain,int(.5*len(remain)))
    test = list(set(remain)-set(valid))

    return(train,valid,test)



def get_names(waste_type, indices):
    '''
    Function to get the filenames for a particular type of trash, given indices.
    - Parameters:  waste_type: the category of waste
                   indices: indices for the train, valid, and test datasets
    - Returns:     file names    
    '''
    file_names = [waste_type+str(i)+".jpg" for i in indices]
    return(file_names)    


def move_files(source_files, destination_folder):
    '''
    Function to move group of source files to another folder.
    Takes as input the list of source files and destination folder.  
    '''
    for file in source_files:
        # Copy files so they still live in the source folder in case we need to repeat / change the operation
        # without having to re-downlaod in the source data
        shutil.copy(file, destination_folder)

        
def split_train_val_test():
    '''
    Function that creates the train, val, and test folders for data (preserving the 
    child folder structure which tells us the waste type). 
    Paths will be train/cardboard, train/glass, etc...  
    '''
    
    subsets = ['train','val']
    waste_types = ['cardboard','glass','metal','paper','plastic','trash', 'compost']
    
    
    inBase = "Documents/GitHub/CS231n-Project-2019/datasets/trashnet/data/dataset-resized"
    inDataPath = os.path.join(userPath, inBase)
    outBase = "Documents/GitHub/CS231n-Project-2019/datasets/trashnet/data/dataset-split"
    outDataPath = os.path.join(userPath, outBase)


    ## create destination folders for data subset and waste type
    for subset in subsets:
        for waste_type in waste_types:
            folder = os.path.join(outDataPath,subset,waste_type)
            if not os.path.exists(folder):
                os.makedirs(folder)

    if not os.path.exists(os.path.join(outDataPath,'test')):
        os.makedirs(os.path.join(outDataPath,'test'))
        
    ## move files to destination folders for each waste type
    for waste_type in waste_types:
        source_folder = os.path.join(inDataPath,waste_type)
        train_ind, valid_ind, test_ind = split_indices(source_folder,1,1)

        ## move source files to train
        train_names = get_names(waste_type,train_ind)
        train_source_files = [os.path.join(source_folder,name) for name in train_names]
        train_dest = userPath.replace("\\", "/") + "/" + outBase + "/train/" + waste_type
        move_files(train_source_files,train_dest)
        
        ## move source files to valid
        valid_names = get_names(waste_type,valid_ind)
        valid_source_files = [os.path.join(source_folder,name) for name in valid_names]
        valid_dest = userPath.replace("\\", "/") + "/" + outBase + "/val/" + waste_type
        move_files(valid_source_files,valid_dest)
        
        ## move source files to test
        test_names = get_names(waste_type,test_ind)
        test_source_files = [os.path.join(source_folder,name) for name in test_names]
        
        ## I use data/test here because the images can be mixed up
        move_files(test_source_files, userPath.replace("\\", "/") + "/" + outBase + "/test")


def augment():
    ''' Function that sets up data augmentation transforms.
    After loading the data into memory, can call this function to get the transforms and apply
    them to the data.
    '''
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }
    return data_transforms


def load_data(outDataPath):
    ''' Function to load the  data from the given path
    Aplies the datatransforms given via augment() and creates and returns
    dataloader objects for the train and val datasets, the sizes of the 
    datasets, and  the list of classnames'''
    
    # Get data transforms
    data_transforms = augment()
    
    # Create an ImageFolder dataloader for the input data
    # See https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(outDataPath, x),
                                              data_transforms[x]) for x in ["train", "val"]}

    # Create DataLoader objects for each of the image datasets returned by ImageFolder
    # See https://pytorch.org/docs/stable/data.html
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    
    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    
    return dataloaders, datasets_sizes, class_names


if __name__ == "__main__":

    ## If this is the file being run as the main method, it creates the train/val/test split folders ##
    split_train_val_test()
    

    ## otherwise use from the vgg.py file to load the data into dataloader objects ##
    #outpath = "/Users/sarahciresi/Documents/GitHub/CS231n-Project-2019/datasets/trashnet/data"  
    #outpath = "/home/sarahciresi/gcloud/project/CS231n-Project-2019/datasets/trashnet/data" 
    
    
    
