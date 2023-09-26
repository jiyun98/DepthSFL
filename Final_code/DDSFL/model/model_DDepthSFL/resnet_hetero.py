import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up

from model.model_DDepthSFL.blocks import *
from model.model_DDepthSFL.resnet18 import *
from model.model_DDepthSFL.resnet34 import *
from model.model_DDepthSFL.resnet50 import *
from model.model_DDepthSFL.resnet56_rev import *
from model.model_DDepthSFL.resnet101 import *
from model.model_DDepthSFL.resnet110 import *
# from blocks import *
# from resnet18 import *
# from resnet34 import *
# from resnet101 import *
# 세 종류의 모델을 가지고 실험

'''resnet18'''
def resnet18_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_client_v1(block = BasicBlock, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_client_v2(block = BasicBlock, num_blocks = [2,0,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_client_v3(block = BasicBlock, num_blocks = [2,2,0,0], num_classes = num_classes)
    return model

def resnet18_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_server_v1(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_server_v2(block = BasicBlock, num_blocks = [0,2,2,2], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_server_v3(block = BasicBlock, num_blocks = [0,0,2,2], num_classes = num_classes)
    return model

'''resnet34'''
def resnet34_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet34_client_v1(block = BasicBlock, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_client_v2(block = BasicBlock, num_blocks = [3,2,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_client_v3(block = BasicBlock, num_blocks = [3,4,2,0], num_classes = num_classes)
    return model

def resnet34_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet34_server_v1(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_server_v2(block = BasicBlock, num_blocks = [0,2,6,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_server_v3(block = BasicBlock, num_blocks = [0,0,4,3], num_classes = num_classes)
    return model


'''resnet50'''
def resnet50_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_client_v1(block = BottleNeck, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_client_v2(block = BottleNeck, num_blocks = [3,2,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_client_v3(block = BottleNeck, num_blocks = [3,4,2,0], num_classes = num_classes)
    return model

def resnet50_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_server_v1(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_server_v2(block = BottleNeck, num_blocks = [0,2,6,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_server_v3(block = BottleNeck, num_blocks = [0,0,4,3], num_classes = num_classes)
    return model

'''resnet56'''
def resnet56_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_client_v1(block = BasicBlock, num_blocks = [9,9,9],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_client_v2(block = BasicBlock, num_blocks = [9,9,9],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_client_v3(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    return model

def resnet56_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_server_v1(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_server_v2(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_server_v3(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    return model


'''resnet101'''
def resnet101_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_client_v1(block = BottleNeck, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_client_v2(block = BottleNeck, num_blocks = [3,4,1,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_client_v3(block = BottleNeck, num_blocks = [3,4,11,0], num_classes = num_classes)
    return model

def resnet101_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_server_v1(block = BottleNeck, num_blocks = [3,4,23,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_server_v2(block = BottleNeck, num_blocks = [0,0,22,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_server_v3(block = BottleNeck, num_blocks = [0,0,12,3], num_classes = num_classes)
    return model

'''resnet110'''
def resnet110_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet110_client_v1(block = BasicBlock, num_blocks = [0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet110_client_v2(block = BasicBlock, num_blocks = [16,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet110_client_v3(block = BasicBlock, num_blocks = [18,12,0], num_classes = num_classes)
    return model

def resnet110_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet110_server_v1(block = BasicBlock, num_blocks = [18,18,18], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet110_server_v2(block = BasicBlock, num_blocks = [2,18,18], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet110_server_v3(block = BasicBlock, num_blocks = [0,6,18], num_classes = num_classes)
    return model

