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
import torch.nn.functional as F
from src.data.augmentations import AugmentationPipeline

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
                 data_files: list,
                 dataset_split: str = 'train',
                 sample_ratio=None,
                 # augmentation
                 augment_config=None,
                 # no. of samples per class for few-shot learning
                 few_shot_k = None,
                 **kwargs):  # Pass through shuffle, etc.

        super().__init__(**kwargs)

        self.root = Path(image_root)
        self.image_root = self.root / image_dir
        self.dataset_split = dataset_split
        # Init augmentation pipeline
        self.augmentation_pipeline = AugmentationPipeline(augment_config or {})

        # Load CSVs
        data_columns = ["img_fp", "label"]
        dfs = [pd.read_csv(f, names=data_columns, header=None) for f in data_files]
        df = pd.concat(dfs, ignore_index=True)
        df["label"] = df["label"].astype(int)

        # Few-shot sampling: sample only K examples per class if few_shot_k is given
        if few_shot_k:
            df = df.groupby("label", group_keys=False).apply(
                lambda x: x.sample(n=min(few_shot_k, len(x)), random_state=42)
            ).reset_index(drop=True)
    
        if sample_ratio:
            df = df.head(int(sample_ratio * len(df)))

        self.num_classes = df['label'].nunique()

        # Only duplicate dataset if augmentations are enabled
        if self.augmentation_pipeline.enabled:
            # Duplicate dataset: Original and Augmented
            df_aug = df.copy()
            df_aug["augmented"] = True
            df["augmented"] = False
            # dataframe is doubled
            self.df = pd.concat([df, df_aug], ignore_index=True).sample(frac=1).reset_index(drop=True)
        else:
            # If augmentation is disabled, don't duplicate
            df["augmented"] = False 
            self.df = df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            row = self.df.iloc[index]
            img_fp, label, is_aug = row['img_fp'], row['label'], row['augmented']
            fname = get_name(img_fp, label)
            img_pil = Image.open(os.path.join(self.root, img_fp)).convert("RGB")
            return_img_pil = img_pil # NOTE: we need to return a PIL of the image too

            # Only on train and augmented samples
            if self.dataset_split == 'train' and is_aug:
                # Apply augmentations
                image = self.augmentation_pipeline.apply(img_pil)
                return_img_pil = self.augmentation_pipeline.get_pil(image)
            else:
                # Resize and normalize all non-augmented samples to ensure consistency
                img_pil = transforms.Resize((224, 224))(img_pil)
                image = self.augmentation_pipeline.to_tensor(img_pil)
                image = self.augmentation_pipeline.normalize(image)
                return_img_pil = self.augmentation_pipeline.get_pil(image)
            
            label = F.one_hot(torch.tensor(int(label)), num_classes=self.num_classes).float()

            # Apply MixUp (only on augmented training samples)
            if self.dataset_split == 'train' and self.augmentation_pipeline.mixup_alpha > 0 and is_aug:
                # Get another sample
                rand_idx = randint(0, self.__len__() - 1)
                row2 = self.df.iloc[rand_idx]
                img_fp2, label2 = row2['img_fp'], row2['label']
                img_pil2 = Image.open(os.path.join(self.root, img_fp2)).convert("RGB")
                
                # Apply augmentations
                image2 = self.augmentation_pipeline.apply(img_pil2)
                label2 = F.one_hot(torch.tensor(int(label2)), num_classes=self.num_classes).float()
                
                # MixUp
                image, label = self.augmentation_pipeline.mixup(image, label, image2, label2)
                return_img_pil = self.augmentation_pipeline.get_pil(image)

            # TODO: Need to get back PIL image of augmented sample
            # TODO: Need to handle tthis for each type of augmentation
            return dict(image=image,
                    image_pil=return_img_pil,
                    label=label,
                    # extra info
                    fns=fname,
                    augmented=is_aug)

        except Exception as e:
            print(f"Skipping index {index} due to error: {e}")
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