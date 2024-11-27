import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Specify dataset and folder to download
dataset_name = "heyitsfahd/nature"
target_folder = "x128/Mountain/"  # Replace with the folder you want from the dataset

# Download path
download_path = "data"
dataset_path = os.path.join(download_path, dataset_name.split('/')[-1])

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
destination_path = "./data/mountains"  # Where you want to move the folder

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