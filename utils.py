
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


def np_to_list(arr):
    this_type_str = type(arr)
    if this_type_str is np.ndarray:
        arr = arr.tolist()
    elif this_type_str in [np.int, np.int32, np.int64]:
        arr = [int(arr), ]
    else:
        arr = arr
    return arr


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")


def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def penalty(x):
    return torch.log(x)


def tpenalty(x, d):
    if x > d:
        return torch.log(x)
