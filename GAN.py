import os
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision import datasets



class Generator(nn.Module):

    def __init__(self):
        self.n = 5



class Discriminator(nn.Module):

    def __init__(self,channels = 3, feature_maps = 64):
        super().__init__()
        self.model = nn.Sequential(

            nn.Conv2d(channels, feature_maps, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(feature_maps, feature_maps*2,kernel_size=4,stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1 (realness score)
            )
        

    def forward(self, x):
        
        output = self.model(x)
        return output.view(-1,1).squeeze(1)
    


        


        



class Trainer():

    def __init__(self):
        self.n = 5


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
    x = 3
    return x

if __name__ == '__main__':
    batch_size = 16
channels = 3
image_size = 64
fake_images = torch.randn(batch_size, channels, image_size, image_size)

# Initialize Discriminator
D = Discriminator(channels=3, feature_maps=64)

# Pass images through Discriminator
outputs = D(fake_images)
print(outputs.shape)   # Should print: torch.Size([16])
print(outputs)      
