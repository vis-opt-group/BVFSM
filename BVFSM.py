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
import function


def train(args, lf=function.lf, uF=function.uF):
    args.y_size = args.y_size
    a = args.a
    b = args.b
    x0 = args.x0
    y0 = args.y0
    d = 1
    seed = 1
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
    dc = args.decay
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

    w = (float(y0) * torch.ones(args.y_size)).cuda().requires_grad_(True)
    h = (float(x0) * torch.ones(args.x_size)).cuda().requires_grad_(True)
    w_z = (float(z0) * torch.ones(args.y_size)).cuda().requires_grad_(True)

    w_opt = torch.optim.SGD([w], lr=y_lr)
    h_opt = torch.optim.Adam([h], lr=x_lr)
    w_z_opt = torch.optim.SGD([w_z], lr=z_lr)

    for x_itr in range(x_loop * TK):
        h_opt.zero_grad()
        step_start_time = time.time()
        lr_decay_rate = 1 / (1 ** (math.floor(x_itr / TK)))
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
        w_opt.param_groups[0]['lr'] = w_opt.defaults['lr'] * lr_decay_rate
        h_opt.param_groups[0]['lr'] = h_opt.defaults['lr'] * lr_decay_rate
        w_z_opt.param_groups[0]['lr'] = w_z_opt.defaults['lr'] * lr_decay_rate

        for z_itr in range(z_loop):
            w_z_opt.zero_grad()

            loss_z = lf(h, w_z) + z_L2 * reg_decay_rate * torch.norm(
                w_z) ** 2
            loss_z.backward()
            w_z_opt.step()

        for y_itr in range(y_loop):
            w_opt.zero_grad()
            loss_w_f = lf(h, w)
            loss_z = lf(h, w_z) + z_L2 * reg_decay_rate * torch.norm(
                w_z) ** 2
            loss_w_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2
            loss_w_ln = y_ln * reg_decay_rate * torch.log(
                loss_w_f.detach() + args.y_size + 1e-8 + loss_z.detach() - loss_w_f)
            loss_w_ = uF(h, w)
            loss_w = loss_w_ + loss_w_L2 - loss_w_ln
            loss_w.backward()
            w_opt.step()

        low_time = time.time() - step_start_time
        hyper_time = time.time()
        h_opt.zero_grad()
        loss_w_f = lf(h, w)
        loss_z = lf(h, w_z) + z_L2 * reg_decay_rate * torch.norm(
            w_z) ** 2
        loss_h_ = uF(h, w)
        loss_h_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2
        loss_h_ln = y_ln * reg_decay_rate * torch.log(loss_w_f.detach() + args.y_size + 1e-8 + loss_z - loss_w_f)
        loss_h = loss_h_ + loss_h_L2 - loss_h_ln


        loss_h.backward()
        h_opt.step()
        step_time = time.time() - step_start_time

        total_time += step_time
        total_hyper_time += (time.time() - hyper_time)

        if x_itr % TK == 0:
            with torch.no_grad():
                loss_test = uF(h, w)

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
                         total_time, hnp[0]] + [ws for ws in utils.np_to_list(wnp)])
