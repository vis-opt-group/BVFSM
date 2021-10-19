import matplotlib
import matplotlib.pyplot as plt
import torch
import hypergrad as hg
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import copy
import time
import csv
import math
import os

import psutil as psutil
import argparse
import utils
import BVFSM


parser = argparse.ArgumentParser()
parser.add_argument('--x_size', type=int, default=1)
parser.add_argument('--y_size', type=int, default=2)
parser.add_argument('--z_loop', type=int, default=50)
parser.add_argument('--y_loop', type=int, default=25)
parser.add_argument('--x_loop', type=int, default=500)
parser.add_argument('--z_lr', type=float, default=0.01)
parser.add_argument('--y_lr', type=float, default=0.01)
parser.add_argument('--x_lr', type=float, default=0.01)
parser.add_argument('--z_L2_reg', type=float, default=0.01)
parser.add_argument('--y_L2_reg', type=float, default=0.01)
parser.add_argument('--y_ln_reg', type=float, default=0.001)
parser.add_argument('--x0', type=float, default=0.)
parser.add_argument('--y0', type=float, default=0.)
parser.add_argument('--a', type=float, default=2.)
parser.add_argument('--b', type=float, default=2.)
parser.add_argument('--c', type=float, default=2.)
parser.add_argument('--decay', type=str, default='log', help='log, power1, power2, poly, linear')

args = parser.parse_args()

a = args.a
b = args.b
C = (float(args.c) * torch.ones(args.y_size)).cuda().requires_grad_(False)


def lf(x, y):
    out = 0
    for i in range(args.y_size):
        out = out + torch.sin((x + y[i] - C[i]))

    return out


def uF(x, y):
    return torch.norm(x - a) ** 2 + torch.norm(y - a - C) ** 2


BVFSM.train(args,lf,uF)



