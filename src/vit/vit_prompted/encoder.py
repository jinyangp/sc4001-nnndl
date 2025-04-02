import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from transformers import AutoModel, AutoImageProcessor, CLIPVisionModelWithProjection, CLIPProcessor

class ImageEncoder(ABC, nn.Module):

    def __init__(self,
                 encoder_processor_name: str,
                 encoder_model_name: str,
                 ):
        
        super().__init__()
        self.encoder_processor_name = encoder_processor_name
        self.encoder_model_name = encoder_model_name

    @abstractmethod
    def preprocess(self, images):
        '''
        To preprocess the image to a shape compatible with the encoder_model using the encoder_processor
        '''
        pass

    @abstractmethod
    def postprocess(self, embeddings):
        '''
        To postprocess the image embeddings from encoder_model into a shape that can be passed on down the pipeline
        '''
        pass


class DINOImageEncoder(ImageEncoder):
    
    def __init__(self,
                 encoder_type,
                 encoder_processor_name,
                 encoder_model_name):
        super().__init__(encoder_processor_name,
                         encoder_model_name)
        self.encoder_type = encoder_type
        self.encoder_processor = AutoImageProcessor.from_pretrained(encoder_processor_name)
        self.encoder_model = AutoModel.from_pretrained(encoder_model_name)

        for param in self.parameters():
            param.requires_grad = False
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # def _tensor_to_pil(self, images):
    #     from torchvision.transforms import ToPILImage
    #     return [ToPILImage()(img) for img in images]

    def preprocess(self, images):
        # pil_images = self._tensor_to_pil(images)
        # return self.encoder_processor(images=pil_images, return_tensors="pt")
        return self.encoder_processor(images=images, return_tensors="pt").to(self.device)
    
    def postprocess(self, embeddings):
        return embeddings.detach()
    
    @torch.no_grad()
    def forward(self, images):
        images = self.preprocess(images)
        self.encoder_model.eval()
        embeds = self.postprocess(self.encoder_model(**images)[0])
        return embeds
