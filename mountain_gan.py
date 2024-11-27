import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Specify dataset and folder to download
dataset_name = "heyitsfahd/nature"
target_folder = "x128/Mountain/"  # Replace with the folder you want from the dataset

# Download path
download_path = "data"
dataset_path = os.path.join(download_path, dataset_name.split('/')[-1])
destination_path = "./data/mountains"  # Where you want to move the folder

if os.path.exists(destination_path):
    print(f"Folder '{target_folder}' already exists in '{destination_path}'. Skipping download and move.")
else:
    # Check if the dataset has already been downloaded
    if not os.path.exists(dataset_path):
        # Download the entire dataset
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print("Dataset downloaded to:", download_path)
    else:
        print("Dataset already downloaded.")

    # List the contents of the dataset directory to understand its structure
    print("Contents of the dataset directory:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for d in dirs:
            print(f"{subindent}{d}/")

    # Path to the specific folder
    source_path = os.path.join(dataset_path, target_folder)

    # Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)

    # Check if the target folder exists in the dataset
    if os.path.exists(source_path):
        # Move the specific folder to your desired location
        shutil.move(source_path, destination_path)
        print(f"Extracted '{target_folder}' to '{destination_path}'")
        
        # Delete the original dataset directory
        shutil.rmtree(dataset_path)
        print(f"Deleted the original dataset directory: {dataset_path}")
    else:
        print(f"Folder '{target_folder}' not found in the dataset.")

# Setting device to GPU if available on Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# Define Generator and Discriminator classes
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

# Normalize Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load the dataset
mountain_data = datasets.ImageFolder(root=destination_path, transform=transform)
dataloader = DataLoader(mountain_data, batch_size=64, shuffle=True)

# Get the resolution of one sample image
sample_image, _ = mountain_data[0]
image_resolution = sample_image.shape[1:]
print(image_resolution)