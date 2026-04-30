import torch 
import torch.nn as nn
import math

def softmax(x):
    """Compute softmax values for each of scroes"""
    e_x = exp(x - max(x, axis=1))
    return e_x / e_x.sum(axis=1)