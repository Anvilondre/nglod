# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import itertools as it

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from lib.datasets import *
from lib.validator.metrics import *

class GeometricValidator(object):
    """Geometric validation; sample 3D points for distance/occupancy metrics."""

    def __init__(self, args, device, net):
        self.args = args
        self.device = device
        self.net = net
        self.num_samples = 100000
        self.set_dataset()

    
    def set_dataset(self):
        """Two datasets; 1) samples uniformly for volumetric IoU, 2) samples surfaces only."""

        # Same as training since we're overfitting
        self.val_dataset = MeshDataset(self.args, num_samples=self.num_samples)
        self.val_data_loader = DataLoader(self.val_dataset, 
                                          batch_size=self.num_samples*len(self.args.sample_mode),
                                          shuffle=False, pin_memory=True, num_workers=4)


    def validate(self, epoch):
        """Geometric validation; sample surface points."""

        ys = []
        preds = []
        # Uniform points metrics
        for n_iter, data in enumerate(self.val_data_loader):
            y = data[1].to(self.device)
            ind = y.abs().squeeze() < 1e-3
            y = y[ind]
            X = data[0].to(self.device)[ind]
            ys.append(y > 0)
            preds.append(self.net(X) > 0)
        
        y = torch.cat(ys).cpu()
        pred = torch.cat(preds).cpu()
        return f1_score(y, pred), len(y), y.sum().item(), pred.sum().item()

