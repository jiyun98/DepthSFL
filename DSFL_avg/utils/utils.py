import math
import numpy as np
# from math import ceil as up
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import copy
import logging
import torch.nn as nn
import random
from datetime import datetime

def set_filename(args):
    if args.method == 'dsfl':
        method_name = 'DFSL'
    elif args.method == 'ddsfl':
        if args.kd_server_opt and args.kd_self_opt:
            method_name = 'KSD_DSFL'
        elif args.kd_server_opt:
            method_name = 'KD_DSFL'
        elif args.kd_self_opt:
            method_name = 'SD_DSFL'
        else:
            method_name = 'D_DSFL'
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = '[' + method_name + ']'+'['+ str(args.data) +']'+ '['+ str(args.model_name) +']' + '['+ str(args.seed) +']' + timestamp
    wandb_name = '[' + method_name + ']'+'['+ str(args.data) +']'+ '['+ str(args.model_name) +']' + '['+ str(args.seed) +']' 
    return name, wandb_name, timestamp, method_name

def up(value):
  return math.ceil(value)

             
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc



def lr_setting(lr):
    new_lr = 0.5 * lr
    return new_lr

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class Concat_models(nn.Module):
    def __init__(self, model_1, model_2):
        super(Concat_models,self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self,x):
        rst_tmp = self.model_1(x)
        rst = self.model_2(rst_tmp)
        return rst
