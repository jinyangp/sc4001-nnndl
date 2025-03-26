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

class PromptedViTBackbone(ViT):

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
    

class PromptedViT(PromptedViTBackbone):
    
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
            # If deep, we replicate the prompt embeddings across all encoder layers (+1 for input layer)
            self.prompt_embeddings = nn.Parameter(prompt_embeddings.repeat(self.num_encoder_layers, 1, 1))
        else:
            self.prompt_embeddings = nn.Parameter(prompt_embeddings)

        self.prompt_dropout = prompt_dropout


    def forward(self, x, *args, **kwargs):

        if self.prompt_depth == "shallow":
            x = self.shallow_incorporate_prompt(x, self.prompt_embeddings[0])
            x = self.model(x)
            return x
        
        # NOTE: Each encoder layer outputs only one output, x
        elif self.prompt_depth == "deep":
            x = self.shallow_incorporate_prompt(x, self.prompt_embeddings[0])
            x = self.model.encoder.layers[0](x)
            for i in range(1, self.num_encoder_layers):
                # STEP: Get the corresponding encoder layer
                encoder_layer = self.model.encoder.layers[i]
                # STEP: Integrate prompt with previous encoder layer output
                prompt_embeds = self.prompt_embeddings[i]
                x = self.deep_incorporate_prompt(x, prompt_embeds)
                # STEP: Forward pass into this encoder layer
                x = encoder_layer(x)
            # get classifier CLS token
            x = x[:,0]
            x = self.heads(x)
            return x


    def configure_optimizers(self):

        lr = self.learning_rate
        params = self.parameters()
        opt = torch.optim.AdamW(params, lr)

        if self.use_scheduler:
            metric_monitored = self.scheduler_config.get("monitor", "val/loss")
            if "monitor" in self.scheduler_config:
                del self.scheduler_config["monitor"]
            self.lr_scheduler = ReduceLROnPlateau(opt, **self.scheduler_config)

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": metric_monitored,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        return opt


class PromptedViTResampler(PromptedViTBackbone):
    
    def __init__(self,
                 num_prompt_tokens:int=4,
                 prompt_depth:str='shallow',
                 prompt_dropout:float=0.0,
                 prompt_resampler_config=None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
    
        self.num_tokens = num_prompt_tokens
        self.prompt_depth = prompt_depth
        self.prompt_dropout = prompt_dropout

        self.hidden_dim = self.model.hidden_dim
        self.num_encoder_layers = len(self.model.encoder.layers) # number of transformer encoder layers


    def get_input(self, batch, bs=None):
 
        # NOTE: We need to pass in the prompt embeddings here based on each image in
        # the batch
        if bs is not None:
            for k in batch.keys():
                batch[k] = batch[k][:bs]

        ret_dict = dict(X=[batch['image']], y=[batch['label']], fn=[batch['fns']])
        img_pils = batch["X_pil"]
        prompt_embeds = []
        
        for img in img_pils:
            # NOTE: our self.prompt_embed_model should take in an image
            # PIL image -> (1, num_tokens, token_dim) if depth == "shallow" else (num_layers, num_tokens, token_dim)
            embeds = self.prompt_embed_model(img)
            if self.prompt_depth == "deep":
                embeds = embeds.repeat(self.num_encoder_layers+1,1,1)
            prompt_embeds.append(embeds)
        # (1, num_tokens, token_dim) if depth == "shallow" else (num_layers, num_tokens, token_dim) -> (B, 1, num_tokens, token_dim) if depth == "shallow" else (B, num_layers, num_tokens, token_dim)
        ret_dict['prompt_embed'] = [torch.stack(prompt_embeds,dim=0)]

        return ret_dict


    def forward(self, x, prompt_embeds=None, *args, **kwargs):

        if self.prompt_depth == "shallow":
            x = self.shallow_incorporate_prompt(x, prompt_embeds[0])
            x = self.model(x)
            return x
        
        # NOTE: Each encoder layer outputs only one output, x
        elif self.prompt_depth == "deep":
            x = self.shallow_incorporate_prompt(x, prompt_embeds[0])
            x = self.model.encoder.layers[0](x)
            for i in range(1, self.num_encoder_layers):
                # STEP: Get the corresponding encoder layer
                encoder_layer = self.model.encoder.layers[i]
                # STEP: Integrate prompt with previous encoder layer output
                embeds = prompt_embeds[i]
                x = self.deep_incorporate_prompt(x, embeds)
                # STEP: Forward pass into this encoder layer
                x = encoder_layer(x)
            # get classifier CLS token
            x = x[:,0]
            x = self.heads(x)
            return x


    def shared_step(self, batch):
        
        loss_dict = {}
        
        # STEP: Get the inputs from the batches
        inps = self.get_input(batch)
        X = torch.cat(inps["X"], 1)
        prompt_embed = torch.cat(inps["prompt_embed"], 1)
        y = torch.cat(inps["y"], 1)  

        # STEP: Calculate loss using get_loss
        y_pred = self.model(X, prompt_embeds=prompt_embed)
        loss = self.get_loss(y_pred, y)
        # STEP: Calculate accuracy
        acc = get_acc(y_pred, y)
        # STEP: Log the items
        log_prefix = "train" if self.training else "val"

        loss_dict.update({f'{log_prefix}/loss': loss})
        loss_dict.update({f'{log_prefix}/acc': acc})

        return loss, loss_dict


    def test_step(self, batch, batch_idx):

        inps = self.get_input(batch)
        X, prompts, y, fns = inps['X'], inps['prompt_embed'], inps['y'], inps['fn'][0]
        X = torch.cat(inps["X"], 1)
        y = torch.cat(inps["y"], 1)
        prompts = torch.cat(inps["prompt_embed"], 1)

        y_pred = self.model(X, prompt_embeds=prompts)
        loss = self.get_loss(y_pred, y).item()
        acc = get_acc(y_pred, y)

        y_pred = torch.argmax(y_pred, dim=1)
        target = torch.argmax(y, dim=1)
        incorrect_indices = torch.nonzero(y_pred != target).squeeze().tolist()
        y_pred = y_pred.tolist()
        target = target.tolist()

        if not isinstance(incorrect_indices, list):
            incorrect_indices = [incorrect_indices]
        
        if len(incorrect_indices) > 0:
            incorrect_samples = [(fns[i], y_pred[i], target[i]) for i in incorrect_indices]
        else:
            incorrect_samples = []

        if self.test_outputs is None:
            self.test_outputs = {
            'loss': [loss],
            'acc': [acc],
            'incorrect_samples': incorrect_samples
            }       
        else:
            self.test_outputs['loss'].append(loss)
            self.test_outputs['acc'].append(acc)
            if len(incorrect_samples) > 0:
                self.test_outputs['incorrect_samples'] = self.test_outputs['incorrect_samples'] + incorrect_samples        