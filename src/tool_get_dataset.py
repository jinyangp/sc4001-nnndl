import os
import torch
import argparse
from pathlib import Path
from torchvision import datasets

def get_dataset(data_directory:str=None):

    '''
    This function downloads the dataset and returns three text files (train_img_fp.txt, val_img_fp.txt, test_img_fp.txt)
    '''

    root_data_dir = os.path.join(os.getcwd(), data_directory)
    if not os.path.exists(root_data_dir):
        os.makedirs(root_data_dir, exist_ok=True)

    full_data_dir = os.path.join(root_data_dir, "flowers-102")
    to_download = True
    if os.path.exists(full_data_dir):
        to_download = False        
    
    train_dataset = datasets.Flowers102(root=root_data_dir, split = "train", download=to_download)
    val_dataset = datasets.Flowers102(root=root_data_dir, split = "val", download=to_download)
    test_dataset = datasets.Flowers102(root=root_data_dir, split = "test", download=to_download)

    train_indices = train_dataset._image_files
    train_labels = train_dataset._labels
    val_indices = val_dataset._image_files
    val_labels = val_dataset._labels
    test_indices = test_dataset._image_files
    test_labels = test_dataset._labels

    with open(os.path.join(root_data_dir, "train_images.txt"), "w") as f:
        for idx in range(len(train_dataset)):
            relative_path = Path(train_indices[idx]).relative_to(root_data_dir).as_posix()  # Convert to relative path
            f.write(f'{str(relative_path)},{train_labels[idx]}' + "\n")

    with open(os.path.join(root_data_dir, "val_images.txt"), "w") as f:
        for idx in range(len(val_dataset)):
            relative_path = Path(val_indices[idx]).relative_to(root_data_dir).as_posix()  # Convert to relative path
            f.write(f'{str(relative_path)},{val_labels[idx]}' + "\n")

    with open(os.path.join(root_data_dir, "test_images.txt"), "w") as f:
        for idx in range(len(test_dataset)):
            relative_path = Path(test_indices[idx]).relative_to(root_data_dir).as_posix()  # Convert to relative path
            f.write(f'{str(relative_path)},{test_labels[idx]}' + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get dataset and train-test-split indices.')
    # Adding arguments
    parser.add_argument('--data_directory', type=str, default='data')
    args = parser.parse_args()

    get_dataset(args.data_directory)