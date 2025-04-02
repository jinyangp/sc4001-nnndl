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
    
    def __getitem__(self, index):
        
        try:
            row = self.df.iloc[index]
            img_fp, label = row['img_fp'], row['label']
            fname = get_name(img_fp,label)

            img_pil = Image.open(os.path.join(self.root, img_fp))
            image = self.data_transforms[self.dataset_split](img_pil)
            # NOTE: we make a one-hot encoded vector here
            label = torch.nn.functional.one_hot(torch.tensor([label]), num_classes=self.num_classes)
            label = label.transpose(0,1).squeeze(1).float()
 
            return dict(image=image,
                        image_pil=img_pil,
                        label=label,
                        fns=fname)
        
        except Exception as e:            
            print(f"Skipping index {index}", e)
            #sys.exit()
            return self.skip_sample(index)
        
   
def custom_collate_fn(batch):
    
    collated_batch = {}
    
    for key in batch[0].keys():
        if key == "image_pil":
            # Handle seg_img_pil: collect PIL images or None
            collated_batch[key] = [sample[key] for sample in batch]
        else:
            # For other keys, stack tensors
            collated_batch[key] = [sample[key] for sample in batch]
            if isinstance(collated_batch[key][0], torch.Tensor):
                collated_batch[key] = torch.stack(collated_batch[key])  # Stack tensors
    
    return collated_batch