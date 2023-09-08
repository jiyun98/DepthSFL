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
    
def fjord_local_model_assignment(ps, model_name, device, num_classes = 10):
    local_models = []
    if model_name == 'resnet18':
        for i in range(len(ps)):
            local_models.append(resnet18_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet34':
        for i in range(len(ps)):
            local_models.append(resnet34_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet50':
        for i in range(len(ps)):
            local_models.append(resnet50_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet56':
        for i in range(len(ps)):
            local_models.append(resnet56_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet101':
        for i in range(len(ps)):
            local_models.append(resnet101_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet110':
        for i in range(len(ps)):
            local_models.append(resnet110_f(num_classes, ps[i]).to(device))

    for model in local_models:
        model.train()

    return local_models   

def fjord_global_model_assignment(ps, model_name, device, num_classes = 10):
    if model_name == 'resnet18':
        glob_model = resnet18_f(num_classes, ps).to(device)
    elif model_name == 'resnet34':
        glob_model = resnet34_f(num_classes, ps).to(device)
    elif model_name == 'resnet50':
        glob_model = resnet50_f(num_classes, ps).to(device)
    elif model_name == 'resnet56':
        glob_model = resnet56_f(num_classes, ps).to(device)
    elif model_name == 'resnet101':
        glob_model = resnet101_f(num_classes, ps).to(device)
    elif model_name == 'resnet110':
        glob_model = resnet110_f(num_classes, ps).to(device)
    glob_model.train()

    return glob_model