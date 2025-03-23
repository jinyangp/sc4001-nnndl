import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from abc import abstractmethod
import pickle
import numpy as np
from einops import rearrange
import json
import re
import pandas as pd

from torchvision import transforms
from random import randint

def get_name(filename, label):
    return 'fn-' + filename + '___' + 'label-' + str(label)  

class Loader(Dataset):
    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    @abstractmethod
    def __getitem__(self, ind):
        pass

class FlowersDataset(Loader):

    def __init__(self,
                 image_root,
                 image_dir,
                 data_files:list,
                 dataset_split:str='train',
                 sample_ratio=None, # ratio of entire dataset to sample from
                 mixup_alpha=0.4, # medium alpha provides balanced mixup (recommended)
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        self.root = Path(image_root)
        self.image_root = self.root/image_dir

        data_columns = ["img_fp", "label"]
        dfs = [pd.read_csv(f, names=data_columns, header=None) for f in data_files]
        self.df = pd.concat(dfs, ignore_index=True)
        self.df["label"] = self.df["label"].astype(int)

        if sample_ratio:
            self.df = self.df.head(int(sample_ratio*len(self.df)))
        
        self.num_classes = self.df['label'].nunique()
        self.dataset_split = dataset_split
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def __len__(self):
        return len(self.df)
    
    def mixup(self, image1, label1, image2, label2):
        # Sample lambda between 0 and 1 from Beta distribution 
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Mix images and labels
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_image, mixed_label
    
    def __getitem__(self, index):
        
        try:
            row = self.df.iloc[index]
            img_fp, label = row['img_fp'], row['label']
            fname = get_name(img_fp,label)

            # Load main image
            img_pil = Image.open(os.path.join(self.root, img_fp))
            image = self.data_transforms[self.dataset_split](img_pil)
            
            # One-hot encode label
            label = torch.nn.functional.one_hot(torch.tensor([label]), num_classes=self.num_classes)
            label = label.transpose(0,1).squeeze(1).float()

            # Apply mixup only during training
            if self.dataset_split == 'train' and self.mixup_alpha > 0:
                # Load second image to mix with
                sample = self.random_sample()
                image2, label2 = sample['image'], sample['label']
                
                img_pil2 = Image.open(os.path.join(self.root, image2))
                image2 = self.data_transforms[self.dataset_split](img_pil2)
                
                label2 = torch.nn.functional.one_hot(torch.tensor([label2]), num_classes=self.num_classes)
                label2 = label2.transpose(0, 1).squeeze(1).float()
                
                # Perform mixup
                image, label = self.mixup(image, label, image2, label2)
                
            return dict(image=image,
                        label=label,
                        fns=fname)
        
        except Exception as e:            
            print(f"Skipping index {index}", e)
            #sys.exit()
            return self.skip_sample(index)