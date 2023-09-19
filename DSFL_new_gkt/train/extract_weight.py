import math
import numpy as np
# from math import ceil as up
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import copy
import logging
import torch.nn as nn

'''BN_layer 구분하지 않은 방법들'''
def extract_submodel_weight_from_global(glob_net, net, model_i):
    parent = net.state_dict()
    f = copy.deepcopy(parent)

    parent_glob = glob_net.state_dict()

    for key in parent.keys():
        f[key] = parent_glob[key]
    return f


def extract_submodel_weight_from_global_ax(glob_net, net, model_i):
    parent = net.state_dict()
    f = copy.deepcopy(parent)

    parent_glob = glob_net.state_dict()

    for key in parent.keys():
        shape = parent[key].shape
        if len(shape) == 2:
            f[key] = parent_glob[key][0:shape[0], 0:shape[1]]
        elif len(shape) == 1:
            f[key] = parent_glob[key][0:shape[0]]
        else:
            f[key] = parent_glob[key]
    return f

'''BN_layer 구분한 방법들'''
def extract_submodel_weight_from_global_bn(glob_net, net, BN_layers,  model_i):
    parent = net.state_dict()
    f = copy.deepcopy(parent)

    parent_glob = glob_net.state_dict()

    for key in parent.keys():
        shape = parent_glob[key].shape
        if len(shape)>1:
            f[key] = parent_glob[key]
        elif len(shape) == 1:
            if key!= 'linear.bias':
                f[key] = BN_layers[model_i][key]
            else:
                f[key] = parent_glob[key]
        else:
            f[key] = BN_layers[model_i][key]
    return f


def extract_submodel_weight_from_global_ax_bn(glob_net, net, model_i):
    parent = net.state_dict()
    f = copy.deepcopy(parent)

    parent_glob = glob_net.state_dict()

    for key in parent.keys():
        shape = parent[key].shape
        if len(shape) == 2:
            f[key] = parent_glob[key][0:shape[0], 0:shape[1]]
        elif len(shape) == 1:
            f[key] = parent_glob[key][0:shape[0]]
        else:
            f[key] = parent_glob[key]
    return f
