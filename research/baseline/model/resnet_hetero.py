import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up

from model.blocks import *
from model.resnet18 import *
from model.resnet34 import *
from model.resnet50 import *
from model.resnet56 import *
from model.resnet101 import *
from model.resnet110 import *
# from blocks import *
# from resnet18 import *
# from resnet34 import *
# from resnet101 import *
# 세 종류의 모델을 가지고 실험

'''resnet18'''
def resnet18_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_client_v1(block = BasicBlock, num_blocks = [2,2,2,2],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_client_v2(block = BasicBlock, num_blocks = [2,2,2,2],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_client_v3(block = BasicBlock, num_blocks = [2,2,2,1], num_classes = num_classes)
    return model

def resnet18_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_server_v1(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_server_v2(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_server_v3(block = BasicBlock, num_blocks = [2,2,2,1], num_classes = num_classes)
    return model

'''resnet34'''
def resnet34_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet34_client_v1(block = BasicBlock, num_blocks = [3,4,6,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_client_v2(block = BasicBlock, num_blocks = [3,4,4,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_client_v3(block = BasicBlock, num_blocks = [3,4,6,1], num_classes = num_classes)
    return model

def resnet34_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet34_server_v1(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_server_v2(block = BasicBlock, num_blocks = [3,4,2,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_server_v3(block = BasicBlock, num_blocks = [3,4,6,2], num_classes = num_classes)
    return model


'''resnet50'''
def resnet50_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_client_v1(block = BottleNeck, num_blocks = [3,4,6,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_client_v2(block = BottleNeck, num_blocks = [3,4,4,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_client_v3(block = BottleNeck, num_blocks = [3,4,6,1], num_classes = num_classes)
    return model

def resnet50_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_server_v1(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_server_v2(block = BottleNeck, num_blocks = [3,4,2,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_server_v3(block = BottleNeck, num_blocks = [3,4,6,2], num_classes = num_classes)
    return model

'''resnet56'''
def resnet56_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_client_v1(block = BasicBlock, num_blocks = [9,1,9],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_client_v2(block = BasicBlock, num_blocks = [9,9,9],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_client_v3(block = BasicBlock, num_blocks = [9,9,4], num_classes = num_classes)
    return model

def resnet56_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_server_v1(block = BasicBlock, num_blocks = [9,8,9], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_server_v2(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_server_v3(block = BasicBlock, num_blocks = [9,9,5], num_classes = num_classes)
    return model


'''resnet101'''
def resnet101_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_client_v2(block = BottleNeck, num_blocks = [3,4,1,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_client_v2(block = BottleNeck, num_blocks = [3,4,9,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_client_v2(block = BottleNeck, num_blocks = [3,4,20,3], num_classes = num_classes)
    return model

def resnet101_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_server_v2(block = BottleNeck, num_blocks = [3,4,22,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_server_v2(block = BottleNeck, num_blocks = [3,4,14,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_server_v2(block = BottleNeck, num_blocks = [3,4,3,3], num_classes = num_classes)
    return model

'''resnet110'''
def resnet110_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet110_client_v1(block = BasicBlock, num_blocks = [18,1,18],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet110_client_v2(block = BasicBlock, num_blocks = [18,18,1],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet110_client_v3(block = BasicBlock, num_blocks = [18,18,8], num_classes = num_classes)
    return model

def resnet110_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet110_server_v1(block = BasicBlock, num_blocks = [18,17,18], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet110_server_v2(block = BasicBlock, num_blocks = [18,18,17], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet110_server_v3(block = BasicBlock, num_blocks = [18,18,10], num_classes = num_classes)
    return model


############### '''FjORD'''###################

def resnet18_f(num_classes = 10, ps = 1):
    return ResNet18_fjord(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = ps, num_classes = num_classes)

def resnet34_f(num_classes = 10, ps = 1):
    return ResNet18_fjord(block = BasicBlock, num_blocks = [3,4,6,3], p_drop = ps, num_classes = num_classes)

def resnet50_f(num_classes = 10, ps = 1):
    return ResNet18_fjord(block = BottleNeck, num_blocks = [3,4,6,3], p_drop = ps, num_classes = num_classes)

def resnet56_f(num_classes = 10, ps = 1):
    return ResNet56_fjord(block = BasicBlock, num_blocks = [9,9,9], p_drop = ps, num_classes = num_classes)

def resnet101_f(num_classes = 10, ps = 1):
    return ResNet18_fjord(block = BottleNeck, num_blocks = [3,4,23,3], p_drop = ps, num_classes = num_classes)

def resnet110_f(num_classes = 10, ps = 1):
    return ResNet56_fjord(block = BasicBlock, num_blocks = [18,18,18], p_drop = ps, num_classes = num_classes)

