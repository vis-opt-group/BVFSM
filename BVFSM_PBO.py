
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


def show_memory_info(hint):
    pid = os.getpid()

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
save_log=True

initial=[0]
for iu in [2]:
    for a,b in zip([2],[2]):
        for x0,y0 in zip([1],[7]):
            d=1
            tSize=2
            # synthetic data generation
            seed = 1
            n = 100
            val_perc = 0.5
            np.random.seed(seed)
            z_loop = 50
            y_loop = 25
            x_loop = 500
            z_L2 = 0.001
            y_L2 = 0.001
            y_ln = 0.7
            z_ln = 0.1
            z_lr = 0.01
            y_lr = 0.01
            x_lr = 0.01
            TK=1
            z0=y0
            decay=['log','power1','power2','poly','linear']
            dc=decay[iu]
            dcr=1.001


            # In[4]:
            total_time = 0
            total_hyper_time = 0
            if save_log:
                log_path = "PBO\\x1_nonconvex_dc{}{}_TK{}_xyz{}{}{}_ab{}{}._{}.csv".format( dc,dcr,TK,x0,y0,z0,a,b, time.strftime("%Y_%m_%d_%H_%M_%S"))
                with open(log_path, 'a', encoding='utf-8', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(
                        ['z_loop{}-y_loop{}-z_L2{}-y_L2{}-y_ln{}-z_lr{}-y_lr{}-x_lr{}'.format(z_loop, y_loop, z_L2,
                                                                                              y_L2, y_ln, z_lr,
                                                                                              y_lr, x_lr),
                         'd','x_itr','test loss', 'h_norm','step_time','total_time','x','y','x+y'])

            use_gpu = False
            Fmin=0
            xmin=[]
            xmind=0
            for i in range(tSize):
                Fmin=Fmin+(-np.pi/4/(i+1)-a)**2+(-np.pi/4/(i+1)-b)**2
                xmin.append(-np.pi/4/(i+1))
                xmind=xmind+(-np.pi/4/(i+1))**2

            w = (float(y0)*torch.ones(tSize)).cuda().requires_grad_(True)
            h =(float(x0)*torch.ones(1)).cuda().requires_grad_(True)
            w_z = (float(y0)*torch.ones(tSize)).cuda().requires_grad_(True)

            w_opt = torch.optim.SGD([w], lr=y_lr)
            h_opt = torch.optim.Adam([h], lr=x_lr)
            w_z_opt = torch.optim.SGD([w_z], lr=z_lr)

            C = (float(2) * torch.ones(tSize)).cuda().requires_grad_(False)
            def lf(x,y):
                out = 0
                for i in range(tSize):
                    out = out + torch.sin((x + y[i] - C[i]))

                # print(out)

                return out
            def uF(x,y):
                return torch.norm(x - a) ** 2 - torch.norm(y - a - C) ** 2
            print(uF(float(-0.42)*torch.ones(1).cuda(),float(7.14)*torch.ones(tSize).cuda()))
            print(lf(float(-0.42)*torch.ones(1).cuda(),float(7.14)*torch.ones(tSize).cuda()))

            # z_loss_min=1e-6
            for x_itr in range(x_loop*TK):
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
                    # print(loss_z_l-loss_z.item())
                    w_z_opt.step()

                loss_y_l = 0
                wl = w
                # print(w_z.item())

                for y_itr in range(y_loop):
                    w_opt.zero_grad()
                    loss_w_f = lf(h,w)
                    loss_z = lf(h,w_z) + z_L2 * reg_decay_rate * torch.norm(
                        w_z) ** 2
                    loss_w_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2

                    loss_w_ln = y_ln *  torch.log(loss_w_f.detach()+tSize+1e-8+loss_z.detach() - loss_w_f)
                    loss_w_ =uF(h,w)


                    loss_w = -loss_w_ + loss_w_L2 - loss_w_ln
                    grad_w_ = torch.autograd.grad(loss_w_, [w], retain_graph=True, allow_unused=True)
                    grad_w_L2 = torch.autograd.grad(loss_w_L2, [w], retain_graph=True, allow_unused=True)
                    grad_w_ln = torch.autograd.grad(loss_w_ln, [w], retain_graph=True, allow_unused=True)

                    loss_w.backward()
                    torch.nn.utils.clip_grad_norm_([w],10)
                    w_opt.step()
                    wl = w
                    yhis.append(w[0].item())


                low_time = time.time() - step_start_time
                hyper_time = time.time()
                h_opt.zero_grad()
                loss_w_f = lf(h,w)
                loss_z = lf(h,w_z) + z_L2 * reg_decay_rate * torch.norm(
                        w_z) ** 2
                loss_h_ = uF(h,w)
                loss_h_L2 = y_L2 * reg_decay_rate * torch.norm(w) ** 2
                loss_h_ln = y_ln  * torch.log(loss_w_f.detach()+tSize+1e-8+ loss_z - loss_w_f)
                loss_h = loss_h_  + loss_h_ln
                grad_h=torch.autograd.grad(loss_h,[h],retain_graph=True)
                grad_h_ = torch.autograd.grad(loss_h_, [h], retain_graph=True, allow_unused=True)
                grad_h_L2 = torch.autograd.grad(loss_h_L2, [h], retain_graph=True, allow_unused=True)
                grad_h_ln = torch.autograd.grad(loss_h_ln, [h], retain_graph=True, allow_unused=True)

                loss_h.backward()
                torch.nn.utils.clip_grad_norm_([h], 100)

                # print(loss_L2([grad_h]))
                h_opt.step()
                step_time = time.time() - step_start_time

                total_time += step_time
                total_hyper_time += (time.time() - hyper_time)

                if x_itr % TK == 0:
                    with torch.no_grad():
                        loss_test = uF(h,w)
                        loss_train = lf(h,w)

                        print(
                            'd={:d},x_itr={:d},F={:.2f},x={:.2f},y={:.2f},x+y={:.2f},gw_={:.2f}, gw_L2={:.2f}, gw_ln=={:.2f},gh={:.2f}, gh_={:.2f}, gh_ln=={:.2f}'.format(
                                d, x_itr, loss_test.data,
                                h.item(),w[0].item(),h.item()+w[0].item(),grad_w_[0][0].item(),grad_w_L2[0][0].item(),grad_w_ln[0][0].item(),grad_h[0].item(),grad_h_[0].item(),grad_h_ln[0].item()))
                        if save_log:
                            with open(log_path, 'a', encoding='utf-8', newline='') as f:
                                csv_writer = csv.writer(f)
                                csv_writer.writerow(
                                    [d, x_itr, loss_test.data, h.norm() / d,
                                                                          step_time,
                                    total_time,h.item(),w[0].item(),h.item()+w[0].item()]+[yh for yh in yhis])
