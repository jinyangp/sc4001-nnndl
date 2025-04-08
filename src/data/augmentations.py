import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

class AugmentationPipeline:
    def __init__(self, config):
        
        # Enable or disable augmentations
        self.enabled = config.get("enabled", True)  
        # apply multi or single augmentations
        self.mode = config.get("mode", "multi") 
        # Load probabilities and hyperparameters from YAML config
        self.prob_flip = config.get("flip_prob", 0.5)
        self.prob_crop = config.get("crop_prob", 0.5)
        self.prob_color = config.get("color_jitter_prob", 0.5)
        self.mixup_alpha = config.get("mixup_alpha", 0.0)

        # Define each transformation
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)  
        self.crop_transform = transforms.RandomCrop(224, padding=4)

       # Final standard conversion and normalization
        self.to_tensor = transforms.ToTensor()
        self.normalise_means = [0.485, 0.456, 0.406]
        self.normalise_stdev = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.normalise_means, self.normalise_stdev)

    def _apply_jitter(self, image):
        
        # Apply all colour alterations except for hue
        colour_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0  # We'll handle hue separately
        )
        image = colour_jitter(image)
        
        image_tensor = TF.pil_to_tensor(image).float() / 255.0 
        # Manually apply hue jitter (clamped to [-0.5, 0.5])
        hue_factor = random.uniform(-0.1, 0.1)
        hue_factor = max(min(hue_factor, 0.5), -0.5)
        image = TF.adjust_hue(image_tensor, hue_factor) # apply this on tensor to prevent overflow
        image = TF.to_pil_image(image_tensor)
        return image

    def apply(self, image):
    
        if not self.enabled:
            # no augmentations applied, just convert to tensor and normalize
            image = transforms.Resize((224, 224))(image)
            image = self.to_tensor(image)
            image = self.normalize(image)
            return image
        
        # Multi: Apply all augmentations
        if self.mode == "multi":
            if random.random() < self.prob_flip:
                image = self.flip_transform(image)
            if random.random() < self.prob_crop:
                image = self.crop_transform(image)
            if random.random() < self.prob_color:
                image = self._apply_jitter(image)

        # Single: randomly choose one augmentation 
        elif self.mode == "single":
            augmentations = [
                (self.flip_transform, self.prob_flip),
                (self.crop_transform, self.prob_crop),
                (self.color_jitter, self.prob_color)
            ]
            valid_transforms = [aug for aug, p in augmentations if random.random() < p]
            if valid_transforms:
                # Randomly choose one valid transform
                chosen_transform = random.choice(valid_transforms)
                image = chosen_transform(image)

        # Final resize to gurantee consistent shape for batching
        image = transforms.Resize((224, 224))(image)
        
        # Always convert to tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image
    
    def get_pil(self, img_tensor):
        '''
        Receives an image tensor and inverses the transformation to get back a PIL image.
        '''
        img_tensor = img_tensor * torch.tensor(self.normalise_stdev).view(-1, 1, 1)
        img_tensor = img_tensor + torch.tensor(self.normalise_means).view(-1, 1, 1)
        img_tensor = torch.clamp(img_tensor, min=0., max=1.)
        img_tensor = img_tensor.float()
        img_pil = transforms.ToPILImage()(img_tensor)
        return img_pil
            

    def mixup(self, image1, label1, image2, label2):
        
        if not self.enabled or self.mixup_alpha <= 0:
            # no MixUp applied, just return the original images and labels
            return image1, label1
        
        # MixUp performed using beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_image, mixed_label