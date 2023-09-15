from blocks import *
from resnet_template import *
from resnet_hetero import *


'''
homogeneous
'''
def resnet18(num_classes):
    model = ResNet(block = BasicBlock,
                   num_blocks = [2,2,2,2],
                   num_classes =  num_classes,
                    )
    return model

def resnet34(num_classes):
    model = ResNet(block = BasicBlock,
                   num_blocks = [3,4,6,3],
                   num_classes =  num_classes,
                    )
    return model

def resnet50(num_classes):
    model = ResNet(block = BottleNeck,
                   num_blocks = [3,4,6,3],
                   num_classes = num_classes)
    return model

def resnet56(num_classes):
    model = ResNet_c(block=BasicBlock,
                   num_blocks=[9, 9, 9],
                   num_classes=num_classes,
                   )
    return model

def resnet101(num_classes):
    model = ResNet(block = BottleNeck,
                   num_blocks = [3,4,23,3],
                   num_classes = num_classes)
    return model

def resnet110(num_classes):
    model = ResNet_c(block=BasicBlock, 
                   num_blocks=[18, 18, 18],
                   num_classes=num_classes,
                   )
    return model

'''
heterogeneous
'''