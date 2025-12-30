import numpy as np
from .math import log
from .specials import logsumexp

def mse_loss(predicted, target):
    return ((predicted - target) ** 2).mean()

def nll_loss(predicted, target):
    correct_probs = predicted[np.arange(predicted.shape[0]), target._array]
    log_prob = log(correct_probs)
    return -log_prob.mean()

def bce_loss(predicted, target):
    return -(target * log(predicted) + (1 - target) * log(1 - predicted)).mean()