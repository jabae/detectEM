from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Model wrapper for training.
    """
    def __init__(self, model, mip=1):
        super(Model, self).__init__()
        self.model = model

    def forward(self, sample):

        mask = self.model(sample['image'])
        
        preds = {}
        preds["mask1"] = torch.sigmoid(mask[:,0,:,:].reshape((1,1,)+mask.shape[2:]))
        preds["mask2"] = torch.sigmoid(mask[:,1,:,:].reshape((1,1,)+mask.shape[2:]))
        
        return preds

    def save(self, fpath):
        torch.save(self.model.state_dict(), fpath)

    def load(self, fpath):
        state_dict = torch.load(fpath)
        
        self.model.load_state_dict(state_dict, strict=False)
