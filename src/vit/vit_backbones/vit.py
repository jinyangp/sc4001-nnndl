import torch
import torchvision
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange, repeat
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from collections import defaultdict

from src.metrics import get_acc
from src.util import instantiate_from_config

class ViT(pl.LightningModule):

    def __init__(self,
                 pretrained_model_weights: str,
                 finetuned_keys: list[str],
                 loss_type: str ='cross_entropy',
                 num_classes: int = 102,
                 scheduler_config=None,
                 *args,
                 **kwargs
                 ):
        
        '''
        Args:
            pretrained_model_ckpt: str, filepath to the checkpoint file
            finetuned_keys:
        '''

        super().__init__()

        self.num_classes = num_classes  
        self.init_from_ckpt(pretrained_model_weights,
                            finetuned_keys=finetuned_keys)
        self.loss_type = loss_type    
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        # NOTE: to store output of batches in test
        self.test_outputs = None
        self.validation_outputs = None


    def init_from_ckpt(self,
                       path,
                       finetuned_keys:list[str] = None,
                       ):
        
        '''
        Args:
            path: str, path to checkpoint file
            ignore_keys: list of keys to ignore
            finetuned_keys: list of keys to finetune
        '''

        # STEP: Load pretrained weights
        self.model = torchvision.models.vit_b_16(weights=path)
        
        # STEP: Fine-tune all selected keys
        for name, param in self.model.named_parameters():
            for ftk in finetuned_keys:
                if ftk in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # STEP: Modify head to fit our number of classes
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, self.num_classes)


    def get_loss(self, pred, target, reduction="mean"):
        if self.loss_type == "cross_entropy":
            # pred: torch.Size([16, 1000]), target: torch.Size([16, 102]) but currently torch.Size([16, 1, 102]
            loss = torch.nn.functional.cross_entropy(pred,target,reduction=reduction)
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
        X, y, fns = inps['X'], inps['y'], inps['fns']

        y_pred = self.model(X)
        loss = self.get_loss(y_pred, y)
        acc = get_acc(y_pred, y)

        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        target = torch.argmax(y, dim=1, keepdim=True)
        incorrect_indices = (torch.nonzero(y_pred != target, as_tuple=True)[0]).tolist()
        incorrect_samples = [(fns[i], y_pred[i], target[i]) for i in incorrect_indices]
        
        if self.test_outputs is None:
            self.test_outputs = {
            'loss': [loss],
            'acc': [acc],
            'incorrect_samples': incorrect_samples
            }       
        else:
            self.test_outputs['loss'].append(loss)
            self.test_outputs['acc'].append(acc)
            self.test_outputs['incorrect_samples'].extend(incorrect_samples)

    # ======================================
    def get_input(self, batch, bs=None):

        if bs is not None:
            for k in batch.keys():
                batch[k] = batch[k][:bs]

        ret_dict = dict(X=[batch['image']], y=[batch['label']], fn=[batch['fns']])
        return ret_dict
        

    def shared_step(self, batch):

        loss_dict = {}
        
        # STEP: Get the inputs from the batches
        inps = self.get_input(batch)
        X = torch.cat(inps["X"], 1)
        # NOTE: Initially y is of shape [B,1]; need to expand to [B,num_cls]
        y = torch.cat(inps["y"], 1)

        # STEP: Calculate loss using get_loss
        y_pred = self.model(X)
        loss = self.get_loss(y_pred, y)
        # STEP: Calculate accuracy
        acc = get_acc(y_pred, y)
        # STEP: Log the items
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
    
        self.validation_outputs = None # resets the output to None for next epoch


    def on_test_epoch_end(self):

        outputs = self.test_outputs
        num_batches = len(outputs)
        ep_loss = sum([tmp['loss'] for tmp in outputs])/num_batches
        ep_acc = sum([tmp['acc'] for tmp in outputs])/num_batches
        incorrect_samples = []
        incorrect_samples = incorrect_samples.extend([tmp['incorrect_samples'] for tmp in outputs])

        print(f'Test loss: {ep_loss}, Test acc: {ep_acc}')
        for sample in incorrect_samples:
            fn, pred, target = sample
            print(f'Filename: {fn} -> Predicted: {pred} | Targeted: {target}')