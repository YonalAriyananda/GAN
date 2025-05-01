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
from torchvision.utils import save_image



class Generator(nn.Module):

    
    def __init__(self, latent_dim=100, channels=3, no_of_fm=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, no_of_fm * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(no_of_fm * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(no_of_fm * 8, no_of_fm * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(no_of_fm * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(no_of_fm * 4, no_of_fm * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(no_of_fm * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(no_of_fm * 2, no_of_fm, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(no_of_fm),
            nn.ReLU(True),

            nn.ConvTranspose2d(no_of_fm, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  
        )

    def forward(self, z):
        output = self.model(z)
        return output




class Discriminator(nn.Module):

    def __init__(self,channels = 3, no_of_fm = 128):
        super().__init__()
        self.model = nn.Sequential(

            nn.Conv2d(channels, no_of_fm, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(no_of_fm, no_of_fm*2,kernel_size=4,stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(no_of_fm * 2),
            nn.LeakyReLU(0.2, inplace=True),

            
            nn.Conv2d(no_of_fm * 2, no_of_fm * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(no_of_fm * 4),
            nn.LeakyReLU(0.2, inplace=True),

           
            nn.Conv2d(no_of_fm * 4, no_of_fm * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(no_of_fm * 8),
            nn.LeakyReLU(0.2, inplace=True),

            
            nn.Conv2d(no_of_fm * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            
            )
        

    def forward(self, x):
        
        output = self.model(x)
        return output.view(-1,1).squeeze(1)
    


        


        







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


    

if __name__ == '__main__':

   epochs = 500
   batch_size = 128   
   image_size = 64
   latent_dim = 100
   lr = 0.0002
     
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   image_directory = r"C:\Users\yonal\Downloads\Prog black 2\GAN\DogPics"  # Make sure this folder has subdirectories with images
   image_size = 64
   batch_size = 16
   num_workers = 2
 
 # Create the Data object
   data_loader_obj = Data(directory=image_directory,
                        image_size=image_size,
                        batch_size=batch_size,
                        num_of_workers=num_workers)
 
 # Get the DataLoader
   dataloader = data_loader_obj.get_dataloader()

   G = Generator(latent_dim=latent_dim,channels=3, no_of_fm=128).to(device)
   D = Discriminator(channels=3,no_of_fm=128).to(device)

   Loss = nn.BCELoss()
   G_optim = optim.Adam(G.parameters(), lr = lr, betas = (0.5,0.999))
   D_optim = optim.Adam(D.parameters(), lr = lr, betas = (0.5,0.999))

   os.makedirs("generated_images", exist_ok=True)

   for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Real and Fake labels
            real_labels = torch.full((batch_size,),0.9, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # === Train Discriminator ===
            outputs_real = D(real_images).view(-1)
            loss_real = Loss(outputs_real, real_labels)

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = G(noise)

            outputs_fake = D(fake_images.detach()).view(-1)
            loss_fake = Loss(outputs_fake, fake_labels)

            loss_D = loss_real + loss_fake

            D_optim.zero_grad()
            loss_D.backward()
            D_optim.step()

            # === Train Generator ===
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = G(noise)
            outputs = D(fake_images).view(-1)

            loss_G = Loss(outputs, real_labels)

            G_optim.zero_grad()
            loss_G.backward()
            G_optim.step()

            # Print Losses occasionally
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

        # Save generated samples after each epoch
        with torch.no_grad():
            fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
            fake_images = G(fixed_noise)
            save_image(fake_images, f"generated_images/fake_images_epoch_{epoch+1}.png", normalize=True)

   print("Training Complete 🚀")