import torch
import torchvision
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict

from src.metrics import get_acc
from src.util import instantiate_from_config

class ResNet101Model(pl.LightningModule):

    def __init__(self,
                 pretrained_model_weights: str,
                 finetuned_keys: list[str] = None,
                 loss_type: str = 'cross_entropy',
                 num_classes: int = 102,
                 scheduler_config=None,
                 *args,
                 **kwargs
                 ):
        
        '''
        Args:
            pretrained_model_weights: str
            finetuned_keys: list of strings representing layer names to finetune
            num_classes: number of classes in the dataset
        '''
        super().__init__()

        self.num_classes = num_classes  
        self.loss_type = loss_type    
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        # Initialize model
        self.init_from_ckpt(pretrained_model_weights, finetuned_keys=finetuned_keys)

        # Storage for outputs during validation and testing
        self.test_outputs = None
        self.validation_outputs = None


    def init_from_ckpt(self, path, finetuned_keys:list[str] = None):
        '''
        Args:
            weights: str, path to checkpoint file
            finetuned_keys: list of keys to finetune
        '''
        # Load pretrained ResNet101
        self.model = torchvision.models.resnet101(weights=path)
        
        # Freeze or unfreeze parameters based on finetuned_keys
        if finetuned_keys:
            for name, param in self.model.named_parameters():
                param.requires_grad = False  # Freeze all parameters by default
                
                # Unfreeze parameters that contain any of the finetuned_keys
                for key in finetuned_keys:
                    if key in name:
                        param.requires_grad = True
                        break
        
        # Modify the final fully connected layer to match our number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Always make sure the final layer is trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True


    def get_loss(self, pred, target, reduction="mean"):
        if self.loss_type == "cross_entropy":
            loss = torch.nn.functional.cross_entropy(pred, target, reduction=reduction)
        return loss
     
    
    def forward(self, x, *args, **kwargs):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
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


    def test_step(self, batch, batch_idx):
        inps = self.get_input(batch)
        X, y, fns = inps['X'], inps['y'], inps['fn'][0]
        X = torch.cat(inps["X"], 1)
        y = torch.cat(inps["y"], 1)

        y_pred = self.model(X)
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

    # ======================================
    def get_input(self, batch, bs=None):
        if bs is not None:
            for k in batch.keys():
                batch[k] = batch[k][:bs]

        ret_dict = dict(X=[batch['image']], y=[batch['label']], fn=[batch['fns']])
        return ret_dict
        

    def shared_step(self, batch):
        loss_dict = {}
        
        # Get the inputs from the batches
        inps = self.get_input(batch)
        X = torch.cat(inps["X"], 1)
        y = torch.cat(inps["y"], 1)

        # Calculate loss using get_loss
        y_pred = self.model(X)
        loss = self.get_loss(y_pred, y)
        # Calculate accuracy
        acc = get_acc(y_pred, y)
        # Log the items
        log_prefix = "train" if self.training else "val"

        loss_dict.update({f'{log_prefix}/loss': loss})
        loss_dict.update({f'{log_prefix}/acc': acc})

        return loss, loss_dict
    
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        if self.validation_outputs is None:
            self.validation_outputs = [loss]
        else:
            self.validation_outputs.append(loss)


    def on_validation_epoch_end(self):
        """Aggregates validation losses and logs epoch-level metrics."""
        outputs = self.validation_outputs
        num_batches = len(outputs)
        avg_val_loss = sum(outputs)/num_batches

        if self.use_scheduler:
            self.lr_scheduler.step(avg_val_loss)
    
        self.validation_outputs = None  # resets the output to None for next epoch


    def on_test_epoch_end(self):
        outputs = self.test_outputs
        print(outputs)

        num_batches = len(outputs['loss'])
        ep_loss = sum([loss_ for loss_ in outputs['loss']])/num_batches
        ep_acc = sum([acc_ for acc_ in outputs['acc']])/num_batches
        incorrect_samples = outputs['incorrect_samples']

        print(f'Test loss: {ep_loss}, Test acc: {ep_acc}')
        for sample in incorrect_samples:
            fn, pred, target = sample
            print(f'Filename: {fn} -> Predicted: {pred} | Targeted: {target}')