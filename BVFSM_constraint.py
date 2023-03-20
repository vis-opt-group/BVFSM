#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
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


def np_to_list(arr):
    this_type_str = type(arr)
    if this_type_str is np.ndarray:
        arr = arr.tolist()
    elif this_type_str in [np.int, np.int32, np.int64]:
        arr = [int(arr),]
    else:
        arr = arr
    return arr

def show_memory_info(hint):
    # 获取当前进程的进程号
    pid = os.getpid()

    # psutil 是一个获取系统信息的库
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")



cuda = False
double_precision = False

default_tensor_str = 'torch.cuda' if cuda else 'torch'
default_tensor_str += '.DoubleTensor' if double_precision else '.FloatTensor'
torch.set_default_tensor_type(default_tensor_str)


def frnp(x):
    t = torch.from_numpy(x).cuda() if cuda else torch.from_numpy(x)
    return t if double_precision else t.float()


def tonp(x, cuda=cuda):
    return x.detach().cpu().numpy() if cuda else x.detach().numpy()


def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss

def penalty(x):
    return torch.log(x)

def tpenalty(x,d):
    if x>d:
        return torch.log(x)
initial=[0,2,4,6,8,10]
# In[3]:
for y_ln in [0.01]:
    for x0,y0 in zip(initial,initial):
        for con1 in range(1):
            z_loop = 50
            y_loop=25
            tSize=2
            a=2
            b=2
            # x0=2
            # y0=2
            # con1=0
            d=1
            iu=2
            # synthetic data generation
            seed = 1
            n = 100
            val_perc = 0.5
            np.random.seed(seed)
            # z_loop = 50
            # y_loop = 25
            x_loop = 500
            z_L2 = 0.01
            y_L2 = 0.01
            # y_ln = 0.01
            z_lr = 0.01
            y_lr = 0.01
            x_lr = 0.01
            TK=1
            # x0=3.
            # y0=3.
            # z0=3.
            # a=0.
            # b=0.
            z0=y0
            decay=['log','power1','power2','poly','linear']
            dc=decay[iu]
            dcr=1.1
            conlist=['log','quad','None']
            con=conlist[con1]


            # In[4]:
            total_time = 0
            total_hyper_time = 0
            log_path = "x1_constraint{}_yln{}_yzloop{}_{}_tSize{}_dc{}{}_TK{}_xyz{}{}{}_ab{}{}._{}.csv".format(con,y_ln,y_loop,z_loop, tSize,dc,dcr,TK,x0,y0,z0,a,b, time.strftime("%Y_%m_%d_%H_%M_%S"))
            with open(log_path, 'a', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    ['z_loop{}-y_loop{}-z_L2{}-y_L2{}-y_ln{}-z_lr{}-y_lr{}-x_lr{}'.format(z_loop, y_loop, z_L2,
                                                                                          y_L2, y_ln, z_lr,
                                                                                          y_lr, x_lr),
                     'd','x_itr','test loss', 'h_norm','step_time','total_time','x','gx','y'])

            # problem definition
            use_gpu = False
            # tSize=2000
            Fmin=0
            xmin=[]
            xmind=0
            for i in range(tSize):
                Fmin=Fmin+(-np.pi/4/(i+1)-a)**2+(-np.pi/4/(i+1)-b)**2
                xmin.append(-np.pi/4/(i+1))
                xmind=xmind+(-np.pi/4/(i+1))**2

            w = (float(0)*torch.ones(tSize)).cuda().requires_grad_(True)
            h =(float(0)*torch.ones(1)).cuda().requires_grad_(True)
            w_z = (float(0)*torch.ones(tSize)).cuda().requires_grad_(True)

            w_opt = torch.optim.Adam([w], lr=y_lr)
            h_opt = torch.optim.Adam([h], lr=x_lr)
            w_z_opt = torch.optim.SGD([w_z], lr=z_lr)

            C = (float(1) * torch.ones(tSize)).cuda().requires_grad_(False)
            def lf(x,y):
                out = 0
                for i in range(tSize):
                    out = out + torch.sin((x + y[i] - C[i]))


                return out
            def uF(x,y):
                return torch.norm(x - a) ** 2 + torch.norm(y - a ) ** 2
            for x_itr in range(x_loop*TK):
                print('-'*50)
                h_opt.zero_grad()
                step_start_time = time.time()
                yhis=[]
                lr_decay_rate = 1 / (1 ** (math.floor(x_itr / TK)))
                if dc=='log':
                    reg_decay_rate = 1 / (math.log(dcr*math.floor((x_itr+1) / TK)))
                elif dc=='power1':
                    reg_decay_rate = 1 / (dcr ** math.floor((x_itr + 1) / TK))
                elif dc=='power2':
                    reg_decay_rate = 1 / (math.floor((x_itr + 1) / TK)**dcr)
                elif dc == 'linear':
                    reg_decay_rate = 1 / (math.floor((x_itr + 1) / TK) * dcr)
                else:
                    assert 1

                w_opt.param_groups[0]['lr'] = w_opt.defaults['lr'] * lr_decay_rate
                h_opt.param_groups[0]['lr'] = h_opt.defaults['lr'] * lr_decay_rate
                w_z_opt.param_groups[0]['lr'] = w_z_opt.defaults['lr'] * lr_decay_rate
                loss_z_l = 0

                for z_itr in range(z_loop):
                    w_z_opt.zero_grad()

                    loss_z = lf(h,w_z) + z_L2 * reg_decay_rate * torch.norm(
                        w_z) ** 2
                    loss_z.backward()
                    w_z_opt.step()
                if x_itr==466:
                    print('y={}'.format(w_z[0].item()))
                loss_y_l = 0
                wl = w

                for y_itr in range(y_loop):

                    w_opt.zero_grad()
                    loss_w_f = lf(h,w)
                    loss_z = lf(h,w_z) + z_L2 * reg_decay_rate * torch.norm(
                        w_z) ** 2
                    loss_w_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2

                    loss_w_ln = y_ln * reg_decay_rate * torch.log(loss_w_f.detach()+tSize+1e-7+loss_z.detach() - loss_w_f)
                    loss_w_ =uF(h,w)

                    if con=='log':
                        ls=(h.detach()+w)*(h.detach()+w-1)
                        # print(-(h.detach()+w)*(h.detach()+w-1)+ls.detach()+1e-1)
                        loss_w = loss_w_ + loss_w_L2 - loss_w_ln- y_ln/ reg_decay_rate *torch.sum(torch.log(-(h.detach()+w)*(h.detach()+w-1)+ls.detach()+2e-1))
                    elif con=='quad':
                        loss_w = loss_w_ + loss_w_L2 - loss_w_ln + y_ln / reg_decay_rate * torch.sum(
                            torch.relu((h.detach() + w) * (h.detach() + w - 1)) ** 2)
                    else:
                        loss_w = loss_w_ + loss_w_L2 - loss_w_ln
                    loss_w.backward()
                    w_opt.step()
                    wl = w

                low_time = time.time() - step_start_time
                hyper_time = time.time()
                h_opt.zero_grad()
                loss_w_f = lf(h,w)
                loss_z = lf(h,w_z) + z_L2 * reg_decay_rate * torch.norm(
                        w_z) ** 2
                loss_h_ = uF(h,w)
                loss_h_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2
                loss_h_ln = y_ln * reg_decay_rate * torch.log(loss_w_f.detach() +tSize+1e-8+ loss_z - loss_w_f)

                if con=='log':
                    ls = (h.detach() + w) * (h.detach() + w - 1)
                    loss_h = loss_h_ + loss_h_L2 - loss_h_ln- y_ln/ reg_decay_rate *torch.sum(torch.log(-(h+w)*(h+w-1)+ls.detach()+2e-1))#+ y_ln/ reg_decay_rate *torch.relu((h)*(h-1))**2
                elif con=='quad':
                    loss_h = loss_h_ + loss_h_L2 - loss_h_ln+ y_ln/ reg_decay_rate *torch.sum(torch.relu((h+w)*(h+w-1)))**2#+ y_ln/ reg_decay_rate *torch.relu((h)*(h-1))**2
                else:
                    loss_h = loss_h_ + loss_h_L2 - loss_h_ln

                grad_h=torch.autograd.grad(loss_h,[h],retain_graph=True)
                grad_h_ = torch.autograd.grad(loss_h_, [h], retain_graph=True, allow_unused=True)
                grad_h_L2 = torch.autograd.grad(loss_h_L2, [h], retain_graph=True, allow_unused=True)
                grad_h_ln = torch.autograd.grad(loss_h_ln, [h], retain_graph=True, allow_unused=True)

                loss_h.backward()
                h_opt.step()
                step_time = time.time() - step_start_time

                total_time += step_time
                total_hyper_time += (time.time() - hyper_time)

                if x_itr % TK == 0:
                    with torch.no_grad():
                        loss_test = uF(h,w)
                        for g in grad_h:
                            gnp=g.detach().cpu().numpy()

                        hnp=h.detach().cpu().numpy()
                        wnp=w.detach().cpu().numpy()

                        print(
                            'd={:d},x_itr={:d},test loss={:.4f}, h_norm={:.4f},step_time={:.4f},total_time={:.4f},x={:.4f}'.format(
                                d, x_itr, loss_test.data, h.norm() / d,
                                                                      step_time,
                                total_time,hnp[0]))
                        with open(log_path, 'a', encoding='utf-8', newline='') as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerow(
                                [d, x_itr, loss_test.data, h.norm() / d,
                                                                      step_time,
                                total_time,hnp[0],gnp[0]]+[ws for ws in np_to_list(wnp)])
