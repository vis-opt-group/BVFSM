# Value-Function-based Sequential Minimization for Bi-level Optimization
This is the official code for the paper ["A Value-Function-based Interior-point Method for Non-convex Bi-level Optimization"](https://icml.cc/virtual/2021/poster/9581) (ICML2021) and its improved version ["Value-Function-based Sequential Minimization for Bi-level Optimization"](https://arxiv.org/abs/2110.04974)

 ""

## Introduction
This is an efficient algorithm for solving bilevel optimization problems. 




## Environment Preparing

Our code runs on the windows platform based on Python 3.6. You can simply execute the following command to automatically configure the environment

```
pip install -r requirement.txt
```

### Training

We provide examples for optimistic BLO.<br>

<div align=center>
  
![optimistic BLO](eq20.png)
</div>

You can adjust the algorithm through parameter setting. We will give the default setting in the following example.
The results will be saved in `./result_{time.strftime("%Y_%m_%d_%H_%M_%S")}.csv`
```
python demo.py
--y_size 2         #Lower level problem dimension
--z_loop 50
--y_loop 25
--x_loop 500
--z_lr 0.01
--y_lr 0.01
--x_lr 0.01
--z_L2_reg 0.01  #\mu
--y_L2_reg 0.01  #\theta
--y_ln_reg 0.001 #\sigma^(1)
--x0 0.
--y0 0.
  ```
  In addition, you can also manually change the lower objective *f* ``def lf(x,y)`` and upper objective *F* ``def uF(x,y)`` in the code to test the functions you need
```
def lf(x, y):
    out = 0
    for i in range(args.y_size):
        out = out + torch.sin((x + y[i] - C[i]))
    return out


def uF(x, y):
    return torch.norm(x - a) ** 2 + torch.norm(y - a - C) ** 2

```
  
  
  
### Reference

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{liu2021value,
	title={A Value-Function-based Interior-point Method for Non-convex Bi-level Optimization},
	author={Liu, Risheng and 
	Liu, Xuan and 
	Yuan, Xiaoming and 
	Zeng, Shangzhi and 
	Zhang, Jin},
	journal={arXiv preprint arXiv:2106.07991},
	booktitle = {ICML},
	year={2021}
}



@misc{liu2021valuefunctionbased,
      title={Value-Function-based Sequential Minimization for Bi-level Optimization}, 
      author={Risheng Liu and Xuan Liu and Shangzhi Zeng and Jin Zhang and Yixuan Zhang},
      year={2021},
      eprint={2110.04974},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
