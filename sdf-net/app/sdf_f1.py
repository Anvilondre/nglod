import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import time

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import moviepy.editor as mpy
from scipy.spatial.transform import Rotation as R
import pyexr

from lib.renderer import Renderer
from lib.models import *
from lib.tracer import *
from lib.options import parse_options
from lib.geoutils import sample_unif_sphere, sample_fib_sphere, normalized_slice

from sklearn.metrics import f1_score

def write_exr(path, data):
    pyexr.write(path, data,
                channel_names={'normal': ['X', 'Y', 'Z'],
                               'x': ['X', 'Y', 'Z'],
                               'view': ['X', 'Y', 'Z']},
                precision=pyexr.HALF)


if __name__ == '__main__':

    # Parse
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--X', type=str, default='./data/X.pt',
                           help='File with points')
    app_group.add_argument('--y', type=str, default='./data/y.pt',
                           help='File with sdfs')
    app_group.add_argument('--eps', type=float, default=1e-3,
                           help='Occupancy epsilon')
    args = parser.parse_args()

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Get model name
    if args.pretrained is not None:
        name = args.pretrained.split('/')[-1].split('.')[0]
    else:
        assert False and "No network weights specified!"

    net = globals()[args.net](args)
    if args.jit:
        net = torch.jit.script(net)

    net.load_state_dict(torch.load(args.pretrained))

    net.to(device)
    net.eval()

    X = torch.load(args.X)
    y = torch.load(args.y)
    ind = y.abs() < args.eps
    y = y[ind] > 0
    X = X[ind]
    yhat = net(X.to(device)) > 0
    print('True distribution: pos:', y.sum().item(), 'neg:', (~y).sum().item())
    print('Predicted distribution: pos:', yhat.sum().item(), 'neg:', (~yhat).sum().item())
    print('F1 score:', f1_score(y.cpu(), yhat.cpu()))
    
    
