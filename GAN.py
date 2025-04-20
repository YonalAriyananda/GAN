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


class Data():         


    def __init__(self,directory,image_size,batch_size,num_of_workers):
        self.image_size = image_size  
        self.batch_size = batch_size
        self.directory  = directory
        self.num_of_workers = num_of_workers 

        self.transform = self.transform_data()
        self.dataset = self.load_data()
        self.dataloader = self.make_dataloader()


    def transform_data(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self):
        return datasets.ImageFolder(root = self.directory, transform = self.transform)
        

    def make_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers = self.num_of_workers
        )

    def get_dataloader(self):
        return self.dataloader


    

def main():
    return x