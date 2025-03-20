import math
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange, repeat
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from collections import defaultdict
from functools import reduce
from operator import mul

from src.metrics import get_acc
from src.util import instantiate_from_config
from src.vit.vit_backbones.vit import ViT

class PromptedViT(ViT):

    def __init__(self,
                 num_prompt_tokens:int=4,
                 prompt_depth:str='shallow',
                 prompt_dropout:float=0.0,
                 *args,
                 **kwargs):
        
        '''
        Args:
            num_prompt_token: int, number of additional tokens to prompt the ViT with
            prompt_depth: str of either ['shallow'/'deep'], if shallow, only inject embeddings at first encoder layer else at every layer
        '''

        super().__init__(*args, **kwargs)
    
        self.num_tokens = num_prompt_tokens
        self.hidden_dim = self.model.hidden_dim
        self.prompt_depth = prompt_depth #
        self.num_encoder_layers = len(self.model.encoder.layers) # number of transformer encoder layers

        prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_dim)) # e.g, (1,4,768)
        val = math.sqrt(6. / float(3 * reduce(mul, (self.model.patch_size, self.model.patch_size), 1) + self.hidden_dim))  # noqa
        # xavier_uniform initialization
        nn.init.uniform_(prompt_embeddings.data, -val, val)

        if self.prompt_depth == "deep":
            # +1 for the initial input to the stack of transformer encoder blocks
            self.prompt_embeddings = prompt_embeddings.repeat(self.num_encoder_layers+1,1,1)
        else:
            self.prompt_embeddings = prompt_embeddings

        self.prompt_dropout = prompt_dropout

    def shallow_incorporate_prompt(self, x, prompt_embeddings):

        '''
        To incorporate prompts at the first transformer encoder layer
        Args:
            x: tensor of shape [B,C,H,W] of the transformed input image
            prompt_embeddings: tensor of shape [B,N,D], of the initial sequence of prompt embeddings
        '''
        
        B = x.shape[0]
        x = self.model._process_input(x) # get the image patches
        batch_class_token = self.model.class_token.expand(B, -1, -1)
        x = torch.cat((
            batch_class_token,
            self.prompt_dropout(prompt_embeddings.expand(B, -1, -1)),
            x
        ), dim=1)

        # shape: (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x
    

    def deep_incorporate_prompt(self, x, prompt_embeddings):
       
        '''
        To incorporate prompts at deeper transformer encoder layer. The aim here is to incorporate prompts of each specific layer to the previous layer's output.
        Args:
            x: tensor of shape [B,CLS+num_tokens+num_img_patches,D], of token sequence length from the previous transformer encoder layer
            prompt_embeddings: tensor of shape [B,N,D], of the initial sequence of prompt embeddings
        '''
        
        B = x.shape[0]
        
        batch_class_token = x[:,:1,:]
        image_patch_tokens = x[:,(1+self.num_tokens):,:]
        x = torch.cat((
            batch_class_token,
            self.prompt_dropout(prompt_embeddings.expand(B, -1, -1)),
            image_patch_tokens
        ), dim=1)

        # shape: (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x
    

    def forward(self, x, *args, **kwargs):

        if self.prompt_depth == "shallow":
            x = self.shallow_incorporate_prompt(x, self.prompt_embeddings[0])
            x = self.model(x)
            return x
        
        # NOTE: Each encoder layer outputs only one output, x
        elif self.prompt_depth == "deep":
            x = self.shallow_incorporate_prompt(x, self.prompt_embeddings[0])
            for i in range(self.num_encoder_layers):
                x = self.model.encoder.layers[i]
                prompt_embeds = self.prompt_embeddings[i+1]
                x = self.deep_incorporate_prompt(x, prompt_embeds)
            # get classifier CLS token
            x = x[:,0]
            x = self.heads(x)
            return x


# TODO: Not yet implemented yet!
# class PromptedViTResampled(ViT):

#     def __init__(self,
#                 num_prompt_tokens:int=4,
#                 token_method:str='random',
#                 token_resampler_config=None,
#                 prompt_depth:str='shallow',
#                 *args,
#                 **kwargs):
    
#         # TODO: Not yet implemented
#         # NOTE: in this method, the tokens are added during the forward pass itself
#         self.resampler = instantiate_from_config(token_resampler_config)
#         raise NotImplementedError
