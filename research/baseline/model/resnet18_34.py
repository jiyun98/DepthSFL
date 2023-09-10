import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# ---------------------------------------------------
#                Vanilla BasicBlock
# ---------------------------------------------------
class BasicBlock(nn.Module):  # Vanilla
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option = 'B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                self.downsample = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

# ---------------------------------------------------
#                split & pruned resnet
# ---------------------------------------------------
class ResNet_client_f_v1(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_client_f_v1, self).__init__()
        self.in_planes = up(64*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        return out
    

class ResNet_server_f_v1(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_server_f_v1, self).__init__()
        self.in_planes = up(64*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2)
        self.linear = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                split & pruned resnet v2
# ---------------------------------------------------
class ResNet_client_f_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_client_f_v2, self).__init__()
        self.in_planes = up(64*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1)
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        return out2

class ResNet_server_f_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_server_f_v2, self).__init__()
        self.in_planes = up(64*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2)
        self.linear = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                split & pruned resnet v3
# ---------------------------------------------------
class ResNet_client_f_v3(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_client_f_v3, self).__init__()
        self.in_planes = up(64*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2)
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        return out3
    

class ResNet_server_f_v3(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_server_f_v3, self).__init__()
        self.in_planes = up(128*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2)
        self.linear = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                split & pruned resnet v4
# ---------------------------------------------------
class ResNet_client_f_v4(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_client_f_v4, self).__init__()
        self.in_planes = up(64*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        return out4
    

class ResNet_server_f_v4(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_server_f_v4, self).__init__()
        self.in_planes = up(256*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2)
        self.linear = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer4(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                split & pruned resnet v5
# ---------------------------------------------------
class ResNet_client_f_v5(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_client_f_v5, self).__init__()
        self.in_planes = up(64*p_drop)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out5
    

class ResNet_server_f_v5(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, block, num_blocks, p_drop, num_classes=10):
        super(ResNet_server_f_v5, self).__init__()
        self.in_planes = up(256*p_drop)
        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.linear = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.avg_pool2d(x, x.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# ---------------------------------------------------
#                ResNet18
# ---------------------------------------------------
# def resnet18_client_f(num_classes = 10, cut_point = 1, p = 1):
#     if cut_point == 1:
#         model = ResNet_client_f_v1(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p, num_classes = num_classes)
#     elif cut_point == 2:
#         model = ResNet_client_f_v2(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p, num_classes = num_classes)
#     elif cut_point == 3:
#         model = ResNet_client_f_v3(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     elif cut_point == 4:
#         model = ResNet_client_f_v4(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     else:
#         model = ResNet_client_f_v5(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     return model
# 
# def resnet18_server_f(num_classes = 10, cut_point = 1, p = 1):
#     if cut_point == 1:
#         model = ResNet_server_f_v1(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     elif cut_point == 2:
#         model = ResNet_server_f_v2(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     elif cut_point == 3:
#         model = ResNet_server_f_v3(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     elif cut_point == 4:
#         model = ResNet_server_f_v4(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     else:
#         model = ResNet_server_f_v5(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
#     return model

# 모델 3개 버전
def resnet18_client_f(num_classes = 10, cut_point = 1, p = 1):
    if cut_point == 1:
        model = ResNet_client_f_v1(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p, num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet_client_f_v2(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p, num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet_client_f_v4(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
    return model

def resnet18_server_f(num_classes = 10, cut_point = 1, p = 1):
    if cut_point == 1:
        model = ResNet_server_f_v1(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet_server_f_v2(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet_server_f_v4(block = BasicBlock, num_blocks = [2,2,2,2], p_drop = p,num_classes = num_classes)
    return model

# ---------------------------------------------------
#                ResNet34
# ---------------------------------------------------
# def resnet34_client_f(num_classes = 10, cut_point = 1, p = 1):
#     if cut_point == 1:
#         model = ResNet_client_f_v1(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p, num_classes = num_classes)
#     elif cut_point == 2:
#         model = ResNet_client_f_v2(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p, num_classes = num_classes)
#     elif cut_point == 3:
#         model = ResNet_client_f_v3(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     elif cut_point == 4:
#         model = ResNet_client_f_v4(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     else:
#         model = ResNet_client_f_v5(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     return model
# 
# def resnet34_server_f(num_classes = 10, cut_point = 1, p = 1):
#     if cut_point == 1:
#         model = ResNet_server_f_v1(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     elif cut_point == 2:
#         model = ResNet_server_f_v2(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     elif cut_point == 3:
#         model = ResNet_server_f_v3(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     elif cut_point == 4:
#         model = ResNet_server_f_v4(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     else:
#         model = ResNet_server_f_v5(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
#     return model

# 모델 3개 버전
def resnet34_client_f(num_classes = 10, cut_point = 1, p = 1):
    if cut_point == 1:
        model = ResNet_client_f_v1(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p, num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet_client_f_v2(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p, num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet_client_f_v4(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
    return model

def resnet34_server_f(num_classes = 10, cut_point = 1, p = 1):
    if cut_point == 1:
        model = ResNet_server_f_v1(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
    elif cut_point == 2:
        model = ResNet_server_f_v2(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
    elif cut_point == 3:
        model = ResNet_server_f_v4(block = BasicBlock, num_blocks = [3, 4, 6, 3], p_drop = p,num_classes = num_classes)
    return model
