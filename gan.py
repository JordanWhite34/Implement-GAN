import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

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
    

# Prepare MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(mnist_data, batch_size=64, shuffle=True)

if __name__ == "__main__":
    latent_dim = 100  # Size of the random noise vector
    data_dim = 28 * 28  # Output size (MNIST images are 28x28)

    generator = Generator(latent_dim, data_dim).to(device)
    hidden_dim = 128  # Number of neurons in hidden layers
    discriminator = Discriminator(data_dim, hidden_dim).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    lr = 0.0002  # Learning rate
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # # Generate random noise to test the Generator
    # z = torch.randn(1, latent_dim)  # A single noise sample
    # fake_data = generator(z)    # Generate fake data
    # print("Fake Data Shape:", fake_data.shape)  # Should be (1, 784)
    # # Assert the Generator's output shape
    # assert fake_data.shape == (1, data_dim), f"Generator output shape mismatch: {fake_data.shape}"

    # Get a batch of real data
    real_data, _ = next(iter(dataloader))  # Ignore labels
    real_data = real_data.view(-1, data_dim)  # Flatten the images
    real_data = real_data.to(device)  # Move to GPU if available

    # Test the Discriminator with real data
    output = discriminator(real_data[:1])  # Use a single sample
    print("Discriminator Output for Real Data:", output)
    # Assert the Discriminator's output is a valid probability
    assert 0 <= output.item() <= 1, f"Discriminator output invalid: {output.item()}"

    # Training parameters
    epochs = 50  # Number of training epochs
    batch_size = 64

    for epoch in range(epochs):
        for real_data, _ in dataloader:
            # Flatten real images
            real_data = real_data.view(-1, data_dim).to(device)
            batch_size = real_data.size(0)

            # Labels for real and fake data
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # Real labels as 0.9 instead of 1.0
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # Fake data -> Label 0.1

            ### Train Discriminator ###
            # Real data loss
            real_output = discriminator(real_data)
            loss_real = criterion(real_output, real_labels)

            # Fake data loss
            z = torch.randn(batch_size, latent_dim, device=device)  # Generate noise
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())  # Detach to avoid backprop into Generator
            loss_fake = criterion(fake_output, fake_labels)

            # Total Discriminator loss
            loss_d = loss_real + loss_fake

            # Backpropagation and optimization
            discriminator_optimizer.zero_grad()
            loss_d.backward()
            discriminator_optimizer.step()

            ### Train Generator ###
            z = torch.randn(batch_size, latent_dim, device=device)  # Generate new noise
            fake_data = generator(z)
            fake_output = discriminator(fake_data)  # No detach, allow backprop
            loss_g = criterion(fake_output, real_labels)  # Fool the Discriminator (label as real)

            # Backpropagation and optimization
            generator_optimizer.zero_grad()
            loss_g.backward()
            generator_optimizer.step()
    

        # Generate and display a grid of images at the end of training
        if epoch == epochs - 1:  # Only display the grid after the last epoch
            z = torch.randn(16, latent_dim, device=device)  # Generate 16 random noise vectors
            generated_data = generator(z).view(-1, 1, 28, 28).cpu()  # Reshape to image format
            
            # Create a grid
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                ax.imshow(generated_data[i].detach().numpy().squeeze(), cmap="gray")
                ax.axis("off")  # Hide axis for cleaner visuals
            
            plt.suptitle("Generated Images After Training", fontsize=16)
            plt.show()

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")