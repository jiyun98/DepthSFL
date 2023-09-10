from model.resnet_depthfl import *
from model.auxnet import Aux_net_v2
import copy

def model_assignment(cut_point, model_name, num_classes, device):
    local_models = []
    auxiliary_models = []
    if model_name == 'resnet18':
        for i in range(len(cut_point)):
                local_models.append(resnet18_client(num_classes, cut_point[i]))
    elif model_name == 'resnet34':
        for i in range(len(cut_point)):
                local_models.append(resnet34_client(num_classes, cut_point[i]))
    elif model_name == 'resnet50':
        for i in range(len(cut_point)):
                local_models.append(resnet50_client(num_classes, cut_point[i]))
    elif model_name == 'resnet56':
        for i in range(len(cut_point)):
                local_models.append(resnet56_client(num_classes, cut_point[i]))
    elif model_name == 'resnet101':
        for i in range(len(cut_point)):
                local_models.append(resnet101_client(num_classes, cut_point[i]))
    elif model_name == 'resnet110':
        for i in range(len(cut_point)):
                local_models.append(resnet110_client(num_classes, cut_point[i]))
    
    '''auxiliary network'''
    for i in range(len(local_models)-1):
        # client-side model
        local_models[i].to(device)   
        local_models[i].train()

        # auxiliary network 
        input = torch.zeros((1,3,32,32)).to(device)  # input image에 따라 달라짐 
        output = local_models[i](input)
        ax_net = Aux_net_v2(output[-1].shape[1], num_classes).to(device)               # <------------------ 종류바꿈
        ax_net.train()
        auxiliary_models.append(ax_net)   

    return copy.deepcopy(local_models),  copy.deepcopy(auxiliary_models)

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
    