
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
parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=2)
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
args = parser.parse_args()


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


tSize = args.size
a = 2
b = 2
x0 = args.x0
y0 = args.y0
d = 1
iu = 2
seed = 1
val_perc = 0.5
np.random.seed(seed)
z_loop = args.z_loop
y_loop = args.y_loop
x_loop = args.x_loop
z_L2 = args.z_L2_reg
y_L2 = args.y_L2_reg
y_ln = args.y_ln_reg
z_lr = args.z_lr
y_lr = args.y_lr
x_lr = args.x_lr
TK = 1
z0 = y0
decay = ['log', 'power1', 'power2', 'poly', 'linear']
dc = decay[iu]
dcr = 1.1
total_time = 0
total_hyper_time = 0
log_path = "result_{}.csv".format(time.strftime("%Y_%m_%d_%H_%M_%S"))
with open(log_path, 'a', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['z_loop{}-y_loop{}-z_L2{}-y_L2{}-y_ln{}-z_lr{}-y_lr{}-x_lr{}'.format(z_loop, y_loop, z_L2,
                                                                              y_L2, y_ln, z_lr,
                                                                              y_lr, x_lr),
         'd', 'x_itr', 'test loss', 'h_norm', 'step_time', 'total_time', 'x', 'y'])

w = (float(y0) * torch.ones(tSize)).cuda().requires_grad_(True)
h = (float(x0) * torch.ones(1)).cuda().requires_grad_(True)
w_z = (float(z0) * torch.ones(tSize)).cuda().requires_grad_(True)

w_opt = torch.optim.SGD([w], lr=y_lr)
h_opt = torch.optim.Adam([h], lr=x_lr)
w_z_opt = torch.optim.SGD([w_z], lr=z_lr)

C = (float(2) * torch.ones(tSize)).cuda().requires_grad_(False)


def lf(x, y):
    out = 0
    for i in range(tSize):
        out = out + torch.sin((x + y[i] - C[i]))

    return out


def uF(x, y):
    return torch.norm(x - a) ** 2 + torch.norm(y - a - C) ** 2


for x_itr in range(x_loop * TK):
    h_opt.zero_grad()
    step_start_time = time.time()
    yhis = []
    if dc == 'log':
        reg_decay_rate = 1 / (math.log(dcr * math.floor((x_itr + 1) / TK)))
    elif dc == 'power1':
        reg_decay_rate = 1 / (dcr ** math.floor((x_itr + 1) / TK))
    elif dc == 'power2':
        reg_decay_rate = 1 / (math.floor((x_itr + 1) / TK) ** dcr)
    elif dc == 'linear':
        reg_decay_rate = 1 / (math.floor((x_itr + 1) / TK) * dcr)
    else:
        assert 1
    loss_z_l = 0

    for z_itr in range(z_loop):
        w_z_opt.zero_grad()

        loss_z = lf(h, w_z) + z_L2 * reg_decay_rate * torch.norm(
            w_z) ** 2
        loss_z.backward()
        w_z_opt.step()

    loss_y_l = 0
    wl = w
    for y_itr in range(y_loop):
        w_opt.zero_grad()
        loss_w_f = lf(h, w)
        loss_z = lf(h, w_z) + z_L2 * reg_decay_rate * torch.norm(
            w_z) ** 2
        loss_w_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2
        loss_w_ln = y_ln * reg_decay_rate * torch.log(
            loss_w_f.detach() + tSize + 1e-8 + loss_z.detach() - loss_w_f)
        loss_w_ = uF(h, w)
        loss_w = loss_w_ + loss_w_L2 - loss_w_ln
        loss_w.backward()
        w_opt.step()
        wl = w

    low_time = time.time() - step_start_time
    hyper_time = time.time()
    h_opt.zero_grad()
    loss_w_f = lf(h, w)
    loss_z = lf(h, w_z) + z_L2 * reg_decay_rate * torch.norm(
        w_z) ** 2
    loss_h_ = uF(h, w)
    loss_h_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2
    loss_h_ln = y_ln * reg_decay_rate * torch.log(loss_w_f.detach() + tSize + 1e-8 + loss_z - loss_w_f)
    loss_h = loss_h_ + loss_h_L2 - loss_h_ln

    # grad_h=torch.autograd.grad(loss_h,[h],retain_graph=True)
    # grad_h_ = torch.autograd.grad(loss_h_, [h], retain_graph=True, allow_unused=True)
    # grad_h_L2 = torch.autograd.grad(loss_h_L2, [h], retain_graph=True, allow_unused=True)
    # grad_h_ln = torch.autograd.grad(loss_h_ln, [h], retain_graph=True, allow_unused=True)

    loss_h.backward()
    h_opt.step()
    step_time = time.time() - step_start_time

    total_time += step_time
    total_hyper_time += (time.time() - hyper_time)

    if x_itr % TK == 0:
        with torch.no_grad():
            loss_test = uF(h, w)
            # for g in grad_h:
            #     gnp=g.detach().cpu().numpy()

            hnp = h.detach().cpu().numpy()
            wnp = w.detach().cpu().numpy()

            print(
                'd={:d},x_itr={:d},test loss={:.4f}, h_norm={:.4f},step_time={:.4f},total_time={:.4f},x={:.4f}'.format(
                    d, x_itr, loss_test.data, h.norm() / d,
                    step_time,
                    total_time, hnp[0]))
            with open(log_path, 'a', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [d, x_itr, loss_test.data, h.norm() / d,
                     step_time,
                     total_time, hnp[0]] + [ws for ws in np_to_list(wnp)])
