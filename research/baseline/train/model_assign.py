from model.resnet_hetero import *
from model.auxnet import Aux_net_v2
import copy

def global_model_assignment(cut_point, model_name, device, num_classes = 10):
    if model_name == 'resnet18':
        net_glob_client = resnet18_client(num_classes, cut_point).to(device)
    elif model_name == 'resnet34':
        net_glob_client = resnet34_client(num_classes, cut_point).to(device)
    elif model_name == 'resnet50':
        net_glob_client = resnet50_client(num_classes, cut_point).to(device)
    elif model_name == 'resnet56':
        net_glob_client = resnet56_client(num_classes, cut_point).to(device)
    elif model_name == 'resnet101':
        net_glob_client = resnet101_client(num_classes, cut_point).to(device)
    elif model_name == 'resnet110':
        net_glob_client = resnet110_client(num_classes, cut_point).to(device)
    net_glob_client.train()

    return net_glob_client    
    
