from model_small.resnet_hetero import *
from model_small.auxnet import Aux_net_v2, Aux_net
import copy


def model_assignment(cut_point, model_name, num_classes, device):
    local_cmodels = []
    local_smodels = []
    auxiliary_models = []
    if model_name == 'resnet18':
        for i in range(len(cut_point)):
                local_cmodels.append(resnet18_client(num_classes, cut_point[i]))
                local_smodels.append(resnet18_server(num_classes, cut_point[i]))
    elif model_name == 'resnet34':
        for i in range(len(cut_point)):
                local_cmodels.append(resnet34_client(num_classes, cut_point[i]))
                local_smodels.append(resnet34_server(num_classes, cut_point[i]))
    elif model_name == 'resnet50':
        for i in range(len(cut_point)):
                local_cmodels.append(resnet50_client(num_classes, cut_point[i]))
                local_smodels.append(resnet50_server(num_classes, cut_point[i]))
    elif model_name == 'resnet56':
        for i in range(len(cut_point)):
                local_cmodels.append(resnet56_client(num_classes, cut_point[i]))
                local_smodels.append(resnet56_server(num_classes, cut_point[i]))
    elif model_name == 'resnet101':
        for i in range(len(cut_point)):
                local_cmodels.append(resnet101_client(num_classes, cut_point[i]))
                local_smodels.append(resnet101_server(num_classes, cut_point[i]))
    elif model_name == 'resnet110':
        for i in range(len(cut_point)):
                local_cmodels.append(resnet110_client(num_classes, cut_point[i]))
                local_smodels.append(resnet110_server(num_classes, cut_point[i]))
    '''auxiliary network'''
    for i in range(len(local_cmodels)):
        # client-side model
        local_cmodels[i].to(device)   
        local_cmodels[i].train()

        # auxiliary network 
        input = torch.zeros((1,3,32,32)).to(device)  # input image에 따라 달라짐 
        output = local_cmodels[i](input)
        ax_net = Aux_net_v2(output[-1].shape[1], num_classes).to(device)           # <-------------------
        ax_net.train()
        auxiliary_models.append(ax_net)   

        # server-side model
        local_smodels[i].to(device)
        local_smodels[i].train()
    return copy.deepcopy(local_cmodels), copy.deepcopy(local_smodels), copy.deepcopy(auxiliary_models)


def global_model_assignment(cut_point, model_name, device, num_classes = 10):
    if model_name == 'resnet18':
        net_glob_client = resnet18_client(num_classes, cut_point[-1]).to(device)
        net_glob_server = resnet18_server(num_classes, cut_point[0]).to(device) 
    elif model_name == 'resnet34':
        net_glob_client = resnet34_client(num_classes, cut_point[-1]).to(device)
        net_glob_server = resnet34_server(num_classes, cut_point[0]).to(device) 
    elif model_name == 'resnet50':
        net_glob_client = resnet50_client(num_classes, cut_point[-1]).to(device)
        net_glob_server = resnet50_server(num_classes, cut_point[0]).to(device) 
    elif model_name == 'resnet56':
        net_glob_client = resnet56_client(num_classes, cut_point[-1]).to(device)
        net_glob_server = resnet56_server(num_classes, cut_point[0]).to(device) 
    elif model_name == 'resnet101':
        net_glob_client = resnet101_client(num_classes, cut_point[-1]).to(device)
        net_glob_server = resnet101_server(num_classes, cut_point[0]).to(device) 
    elif model_name == 'resnet110':
        net_glob_client = resnet110_client(num_classes, cut_point[-1]).to(device)
        net_glob_server = resnet110_server(num_classes, cut_point[0]).to(device) 
    net_glob_client.train()
    net_glob_server.train()

    return copy.deepcopy(net_glob_client), copy.deepcopy(net_glob_server)