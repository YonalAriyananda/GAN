import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder



class Generator(nn.module):



class Discriminator(nn.module):



class Trainer():


class Data():         # Class that contains the functionality for loading and normalising the data ready for training the model.


    def __init__(self,directory,image_size,batch_size,num_of_workers):
        self.image_size = image_size  # standard image size required.
        self.batch_size = batch_size  # batch size( number of samples that the model processes at once before updating the weights during training.)
        self.directory  = directory   # root directory of the images.
        self.num_of_workers = num_of_workers # how many sub processes to use at a time, enables operations to run in parallel.

        self.transform = self.transform_data()   # see below
        self.dataset = self.load_data()          # see below
        self.dataloader = self.make_dataloader() # see below


    def transform_data(self): # Function to transform the data into the nessecary format for smooth operation in the GAN.
        return transforms.Compose([
            transforms.Resize(self.image_size),     # Resizes all the images to the required size.
            transforms.CenterCrop(self.image_size), # rops from the center of the image to the image size.
            transforms.ToTensor(),                  # Converts images into tensors.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes the images pixel values.
        ])

    def load_data(self): # Fucntion to load the data from the dataset into a variable.
        return datasets.ImageFolder(root = self.directory, transform = self.transform)
        

    def make_dataloader(self): # Creates the data loader(model that feeds batches of data to the transform funtion then feeds it to the model for training).
        return DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers = self.num_of_workers
        )

    def get_dataloader(self):  # Getter method for the DataLoader.
        return self.dataloader


    

def main():
    return x