import random
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class AugmentationPipeline:
    def __init__(self, config):
        # Load probabilities and hyperparameters from YAML config
        self.prob_flip = config.get("flip_prob", 0.5)
        self.prob_crop = config.get("crop_prob", 0.5)
        self.prob_color = config.get("color_jitter_prob", 0.5)
        self.mixup_alpha = config.get("mixup_alpha", 0.0)

        # Define each transformation
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)  
        self.crop_transform = transforms.RandomCrop(224, padding=4)
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

       # Final standard conversion and normalization
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def apply(self, image):
        # Apply each augmentation based on config probabilities

        if random.random() < self.prob_flip:
            image = self.flip_transform(image)

        if random.random() < self.prob_crop:
            image = self.crop_transform(image)

        if random.random() < self.prob_color:
            image = self.color_jitter(image)

        # Final resize to gurantee consistent shape for batching
        image = transforms.Resize((224, 224))(image)
        
        # Always convert to tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image

    def mixup(self, image1, label1, image2, label2):
        # MixUp performed using beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_image, mixed_label