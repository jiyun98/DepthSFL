import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up

from model.blocks import *
from model.resnet18 import *
from model.resnet34 import *
from model.resnet101 import *
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
        model = ResNet34_client_v1(block = BottleNeck, num_blocks = [3,4,6,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_client_v2(block = BottleNeck, num_blocks = [3,4,4,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_client_v3(block = BottleNeck, num_blocks = [3,4,6,1], num_classes = num_classes)
    return model

def resnet50_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet34_server_v1(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_server_v2(block = BottleNeck, num_blocks = [3,4,2,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_server_v3(block = BottleNeck, num_blocks = [3,4,6,2], num_classes = num_classes)
    return model

'''resnet56'''
 

'''resnet101'''
def resnet101_client(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_client_v1(block = BottleNeck, num_blocks = [3,4,1,3],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_client_v2(block = BottleNeck, num_blocks = [3,4,9,3],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_client_v3(block = BottleNeck, num_blocks = [3,4,20,3], num_classes = num_classes)
    return model

def resnet101_server(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_server_v1(block = BottleNeck, num_blocks = [3,4,22,3], num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_server_v2(block = BottleNeck, num_blocks = [3,4,14,3], num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_server_v3(block = BottleNeck, num_blocks = [3,4,3,3], num_classes = num_classes)
    return model

