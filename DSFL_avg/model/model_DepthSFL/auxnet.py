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

# ---------------------------------------------------
#               Auxiliary network
# ---------------------------------------------------
class Aux_net(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, dim,  num_classes=10):
        super(Aux_net, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, num_classes)# int(dim*5))

        self.apply(_weights_init)
    def forward(self, x):   # 32, 26, 8 , 8
        out = F.avg_pool2d(x, x.size()[3])  # 32, 26, 1, 1
        out = out.view(out.size(0), -1) # 32, 26
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
class Concat_models(nn.Module):
    def __init__(self, model_1, model_2):
        super(Concat_models, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
    def forward(self,x):
        rst_tmp = self.model_1(x)
        rst = self.model_2(rst_tmp)
        return rst_tmp, rst

class Aux_net_v2(nn.Module):  # Dropout (or pruned) ResNet [width] 
    def __init__(self, dim,  num_classes=10):
        super(Aux_net_v2, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, int(dim/2))
        self.bn1 = nn.BatchNorm1d(num_features=int(dim/2)) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int(dim/2), num_classes)

        self.apply(_weights_init)
    def forward(self, x):   # 32, 26, 8 , 8
        out = F.avg_pool2d(x, x.size()[3])  # 32, 26, 1, 1
        out = out.view(out.size(0), -1) # 32, 26
        out = self.relu(self.bn1(self.linear(out)))
        logits = self.linear2(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
# -----------------------------

class Concat_models(nn.Module):
    def __init__(self, model_1, model_2):
        super(Concat_models, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
    def forward(self,x):
        rst_tmp = self.model_1(x)
        rst = self.model_2(rst_tmp)
        return rst_tmp, rst