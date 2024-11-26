import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Specify dataset and folder to download
dataset_name = "heyitsfahd/nature"
target_folder = "data/Nature/x128/Mountain"  # Replace with the folder you want from the dataset

# Download the entire dataset
download_path = "data"
api.dataset_download_files(dataset_name, path=download_path, unzip=True)
print("Dataset downloaded to:", download_path)

# Path to the specific folder
dataset_path = os.path.join(download_path, dataset_name.split('/')[-1])
source_path = os.path.join(dataset_path, target_folder)
destination_path = "./data/mountains"  # Where you want to move the folder

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