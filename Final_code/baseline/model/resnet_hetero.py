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
def resnet18_FL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_FL_v1(block = BasicBlock, num_blocks = [2,2,2,2],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_FL_v2(block = BasicBlock, num_blocks = [2,2,2,2],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_FL_v3(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    else:
        model = ResNet18_FL(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    return model

'''resnet34'''
def resnet34_FL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet34_FL_v1(block = BasicBlock, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_FL_v2(block = BasicBlock, num_blocks = [3,2,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_FL_v3(block = BasicBlock, num_blocks = [3,4,2,0], num_classes = num_classes)
    else:
        model = ResNet34_FL(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    return model


'''resnet50'''
def resnet50_FL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_FL_v1(block = BottleNeck, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_FL_v2(block = BottleNeck, num_blocks = [3,2,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_FL_v3(block = BottleNeck, num_blocks = [3,4,2,0], num_classes = num_classes)
    else:
        model = ResNet50_FL(block = BottleNeck, num_blocks = [3,4,6,3], num_classes = num_classes)
    return model


'''resnet56'''
def resnet56_FL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_FL_v1(block = BasicBlock, num_blocks = [2,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_FL_v2(block = BasicBlock, num_blocks = [9,9,1],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_FL_v3(block = BasicBlock, num_blocks = [9,9,4], num_classes = num_classes)
    else:
        model = ResNet56_FL(block = BasicBlock, num_blocks = [9,9,9], num_classes = num_classes)
    return model

'''resnet101'''
def resnet101_FL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_FL_v1(block = BottleNeck, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_FL_v2(block = BottleNeck, num_blocks = [3,4,1,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_FL_v3(block = BottleNeck, num_blocks = [3,4,11,0], num_classes = num_classes)
    else:
        model = ResNet101_FL(block = BottleNeck, num_blocks = [3,4,23,3], num_classes = num_classes)
    return model



'''resnet110'''
def resnet110_FL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet110_FL_v1(block = BasicBlock, num_blocks = [0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet110_FL_v2(block = BasicBlock, num_blocks = [16,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet110_FL_v3(block = BasicBlock, num_blocks = [18,12,0], num_classes = num_classes)
    else:
        model = ResNet110_FL(block = BasicBlock, num_blocks = [18,18,18], num_classes = num_classes)
    return model



############### '''FjORD'''###################

def resnet18_f(num_classes = 10, ps = 1):
    return ResNet18_fjord(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = ps, num_classes = num_classes)

def resnet34_f(num_classes = 10, ps = 1):
    return ResNet34_fjord(block = BasicBlock, num_blocks = [3,4,6,3], p_drop = ps, num_classes = num_classes)

def resnet50_f(num_classes = 10, ps = 1):
    return ResNet50_fjord(block = BottleNeck_f, num_blocks = [3,4,6,3], p_drop = ps, num_classes = num_classes)

def resnet56_f(num_classes = 10, ps = 1):
    return ResNet56_fjord(block = BasicBlock, num_blocks = [9,9,9], p_drop = ps, num_classes = num_classes)

def resnet101_f(num_classes = 10, ps = 1):
    return ResNet101_fjord(block = BottleNeck_f, num_blocks = [3,4,23,3], p_drop = ps, num_classes = num_classes)

def resnet110_f(num_classes = 10, ps = 1):
    return ResNet110_fjord(block = BasicBlock, num_blocks = [18,18,18], p_drop = ps, num_classes = num_classes)


############### '''DetpHFL'''###################
def resnet18_depthFL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet18_depthFL_v1(block = BasicBlock, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet18_depthFL_v2(block = BasicBlock, num_blocks = [2,0,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet18_depthFL_v3(block = BasicBlock, num_blocks = [2,2,0,0], num_classes = num_classes)
    else:
        model = ResNet18_depthFL(block = BasicBlock, num_blocks = [2,2,2,2], num_classes = num_classes)
    return model

def resnet34_depthFL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet34_depthFL_v1(block = BasicBlock, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet34_depthFL_v2(block = BasicBlock, num_blocks = [3,2,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet34_depthFL_v3(block = BasicBlock, num_blocks = [3,4,2,0], num_classes = num_classes)
    else:
        model = ResNet34_depthFL(block = BasicBlock, num_blocks = [3,4,6,3], num_classes = num_classes)
    return model

def resnet50_depthFL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet50_depthFL_v1(block = BottleNeck, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet50_depthFL_v2(block = BottleNeck, num_blocks = [3,2,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet50_depthFL_v3(block = BottleNeck, num_blocks = [3,4,2,0], num_classes = num_classes)
    else:
        model = ResNet50_depthFL(block = BottleNeck, num_blocks = [3,4,6,3],  num_classes = num_classes)
    return model

def resnet56_depthFL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet56_depthFL_v1(block = BasicBlock, num_blocks = [2,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet56_depthFL_v2(block = BasicBlock, num_blocks = [9,9,1],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet56_depthFL_v3(block = BasicBlock, num_blocks = [9,9,4], num_classes = num_classes)
    else:
        model = ResNet56_depthFL(block = BasicBlock, num_blocks = [9,9,9],  num_classes = num_classes)
    return model

def resnet101_depthFL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet101_depthFL_v1(block = BottleNeck, num_blocks = [0,0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet101_depthFL_v2(block = BottleNeck, num_blocks = [3,4,1,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet101_depthFL_v3(block = BottleNeck, num_blocks = [3,4,4,0], num_classes = num_classes)
    else:
        model = ResNet101_depthFL(block = BottleNeck, num_blocks = [3,4,23,3],  num_classes = num_classes)
    return model

def resnet110_depthFL(num_classes = 10, cut_point = 1):
    if cut_point == 1:
        model = ResNet110_depthFL_v1(block = BasicBlock, num_blocks = [0,0,0],  num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet110_depthFL_v2(block = BasicBlock, num_blocks = [16,0,0],  num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet110_depthFL_v3(block = BasicBlock, num_blocks = [18,12,0], num_classes = num_classes)
    else:
        model = ResNet110_depthFL(block = BasicBlock, num_blocks = [18,18,18],  num_classes = num_classes)
    return model
