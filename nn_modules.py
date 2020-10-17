import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# def optimize(lr):
#     pass


# def compute_loss():
#     pass


# class LogisticRegressionHomeGrown(nn.module):

#     def __init__(self):
#         # weights
#         # biases

#     def forward(self, X):
#         pass

#     def backward(self):
#         # return gradient
#         pass


class LogisticRegressionLazy(nn.module):

    def __init__(self, nx):
        self.scorer = nn.linear(nx, 1)

    def forward(self, X):
        '''
        X has shape (B, nx)
        '''
        # shape (B, 1)
        z = self.scorer(X)
        # shape (B, 1)
        a = F.sigmoid(z)
        return z, a