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
output_dir = "./generated_images/"

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
    os.makedirs(output_dir, exist_ok=True)

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
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: N x input_dim x 1 x 1
            nn.ConvTranspose2d(input_dim, feature_maps * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),
            # State: (feature_maps*16) x 4 x 4

            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # State: (feature_maps*8) x 8 x 8

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (feature_maps*4) x 16 x 16

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (feature_maps*2) x 32 x 32

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State: (feature_maps) x 64 x 64

            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: img_channels x 128 x 128
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Add a global average pooling layer

    def forward(self, x):
        x = self.model(x)          # Output shape: [batch_size, 1, 13, 13]
        x = self.pool(x)           # Output shape: [batch_size, 1, 1, 1]
        x = x.view(x.size(0), -1)  # Reshape to [batch_size, 1]
        return x
    

# Normalize Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # For RGB
])

# Load the dataset
mountain_data = datasets.ImageFolder(root=destination_path, transform=transform)
dataloader = DataLoader(mountain_data, batch_size=64, shuffle=True)

# Get the resolution of one sample image
sample_image, _ = mountain_data[0]
image_resolution = sample_image.shape[1:]
print(image_resolution)

# Main loop
if __name__ == "__main__":
    # Define training parameters
    epochs = 50
    batch_size = 64
    latent_dim = 100  # Size of the random noise vector
    img_size = 128
    img_channels = 3  # RGB images

    # Labels for real / fake data
    real_label = 0.9
    fake_label = 0.1

    # Initialize models
    generator = Generator(latent_dim, img_channels=img_channels).to(device)
    discriminator = Discriminator(img_channels).to(device)
    hidden_dim = 128  # Number of neurons in hidden layers

    # Optimizers and loss function
    criterion = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            # Move real images to the device
            real_images = real_images.to(device)

            ## Train Discriminator
            # Real Images
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)

            # Forward pass for real images
            real_output = discriminator(real_images)
            loss_real = criterion(real_output, real_labels)

            # Fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)    # Generate random noise
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())   # Detach to avoid training the Generator
            loss_fake = criterion(fake_output, fake_labels)

            # Combine losses and backpropagate
            loss_discriminator = loss_real + loss_fake
            discriminator_optimizer.zero_grad()
            loss_discriminator.backward()
            discriminator_optimizer.step()

            ## Train Generator
            # Generate new fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            # Get discriminator's predictions
            output = discriminator(fake_images)
            loss_generator = criterion(output, real_labels) # Generator wants to fool the discriminator

            # Backpropagate and optimize
            generator_optimizer.zero_grad()
            loss_generator.backward()
            generator_optimizer.step()

            # Print progress
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], "
                    f"Loss D: {loss_discriminator.item():.4f}, Loss G: {loss_generator.item():.4f}")

        # Save images at the end of each epoch
        with torch.no_grad():
            z = torch.randn(16, latent_dim, 1, 1, device=device)  # Generate 16 random noise vectors
            generated_images = generator(z)  # Output shape: [16, 3, 128, 128]
            generated_images = generated_images * 0.5 + 0.5  # Denormalize to [0, 1]

            # Save to file
            save_image(generated_images, f"{output_dir}/epoch_{epoch+1}.png", nrow=4, normalize=True)

        print(f"Generated images saved for epoch {epoch+1} at '{output_dir}/epoch_{epoch+1}.png'.")
    
    # Save final results
    print("Training complete. Check generated images in the 'generated_images' folder.")