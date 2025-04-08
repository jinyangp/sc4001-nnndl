import torch
import torchvision
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from torchvision.models.densenet import DenseNet, _DenseBlock, _DenseLayer

from src.cnn.densenet import DenseNet121Model
from src.metrics import get_acc
from src.util import instantiate_from_config

class _DilatedDenseBlock(_DenseBlock):

    """Custom DenseBlock that supports dilation in the final block"""
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
        dilation: int = 1,
    ) -> None:
        super(_DenseBlock, self).__init__(
            num_layers, num_input_features, bn_size, 
            growth_rate, drop_rate, memory_efficient
        )
        
        # Override the layers with dilated versions
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            
            # Add dilation to the 3x3 conv in _DenseLayer
            # DenseLayer has conv1 (1x1) and conv2 (3x3)
            layer.conv2.stride = (1, 1)
            layer.conv2.padding = (dilation, dilation)
            layer.conv2.dilation = (dilation, dilation)
            
            self.add_module("denselayer%d" % (i + 1), layer)

class DilatedDenseNet121Model(DenseNet121Model):

    def __init__(self,
                 pretrained_model_weights,
                 dilation: int = 2,
                 finetuned_keys = None,
                 loss_type = 'cross_entropy',
                 num_classes = 102,
                 scheduler_config=None,
                 *args,
                 **kwargs):
        self.dilation = dilation
        super().__init__(*args, **kwargs)

    def init_from_ckpt(self, path, finetuned_keys = None):
        
        # STEP: Load in original weights
        self.model = torchvision.models.densenet121(weights=path)

        # STEP: Initialise our new block and replace the old one
        # Get the original denseblock4
        original_denseblock4 = self.model.features.denseblock4

        num_layers = len(original_denseblock4)  # number of DenseLayers in block 4
        growth_rate = self.model.growth_rate
        drop_rate = self.model.drop_rate if hasattr(self.model, "drop_rate") else 0  # fallback if not set
        bn_size = original_denseblock4[0].bn_size if hasattr(original_denseblock4[0], "bn_size") else 4
        transition3_out = self.model.features.transition3
        num_input_features = transition3_out.conv.out_channels
        
        dilated_block = _DilatedDenseBlock(num_layers=num_layers,
                                           num_input_features=num_input_features,
                                           bn_size=bn_size,
                                           growth_rate=growth_rate,
                                           drop_rate=drop_rate,
                                           memory_efficient=False,
                                           dilation = self.dilation
                                           )
        self.model.features.denseblock4 = dilated_block

        # STEP: Change all keys to be finetuned
        if finetuned_keys:
            for name, param in self.model.named_parameters():
                param.requires_grad = False  # Freeze all parameters by default
                
                # Unfreeze parameters that contain any of the finetuned_keys
                for key in finetuned_keys:
                    if key in name:
                        param.requires_grad = True
                        break

        # STEP: Change the classifer head
        # Modify the final fully connected layer to match our number of classes
        # DenseNet uses classifier instead of fc for its final layer
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, self.num_classes)
        
        # Always make sure the final layer is trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True