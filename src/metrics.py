import torch
import torch.nn as nn
import numpy as np

def get_acc(pred, target):

    '''
    Args:
        pred: tensor of shape (B, num_cls)
        target: tensor of shape (B, num_cls)
    '''

    N = pred.shape[0]
    # (B, num_cls) -> (B,1)
    pred = torch.argmax(pred, dim=1, keepdim=True)
    # (B, num_cls) -> (B,1)
    target = torch.argmax(pred, dim=1, keepdim=True)
    num_correct = torch.sum(pred.squeeze() == target.squeeze()).item()  # Remove extra dimension
    return num_correct/N