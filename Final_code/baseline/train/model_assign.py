from model.resnet_hetero import *
from model.auxnet import Aux_net
import copy

'''Exclusive learning 과 global reduction에 사용'''
# FL인데 크기가 줄어든 FL입니다
def FL_model_assignment(cut_point, model_name, device, num_classes = 10):
    cut_points = [1,2,3]
    if model_name == 'resnet18':    # 확인
        net_glob = resnet18_FL(num_classes, cut_point).to(device)
    elif model_name == 'resnet34':  # 확인
        net_glob = resnet34_FL(num_classes, cut_point).to(device)
    elif model_name == 'resnet50':
        net_glob = resnet50_FL(num_classes, cut_point).to(device)
    elif model_name == 'resnet56':
        net_glob = resnet56_FL(num_classes, cut_point).to(device)
    elif model_name == 'resnet101':
        net_glob = resnet101_FL(num_classes, cut_point).to(device)
    elif model_name == 'resnet110':
        net_glob = resnet110_FL(num_classes, cut_point).to(device)
    net_glob.train()

    return net_glob, cut_points   
#==========================================================================
'''FjORD에 사용'''
def fjord_local_model_assignment(model_name, device, num_classes = 10):
    local_models = []
    if model_name == 'resnet18':
        ps = [0.1,0.4,0.5,1.0]
        for i in range(len(ps)):
            local_models.append(resnet18_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet34':
        ps = [0.1,0.4,0.5,1.0]
        for i in range(len(ps)):
            local_models.append(resnet34_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet50':
        ps = [0.1,0.4,0.5,1.0]
        for i in range(len(ps)):
            local_models.append(resnet50_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet56':
        ps = [0.1,0.4,0.5,1.0]
        for i in range(len(ps)):
            local_models.append(resnet56_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet101':
        ps = [0.1,0.4,0.5,1.0]
        for i in range(len(ps)):
            local_models.append(resnet101_f(num_classes, ps[i]).to(device))
    elif model_name == 'resnet110':
        ps = [0.1,0.4,0.5,1.0]
        for i in range(len(ps)):
            local_models.append(resnet110_f(num_classes, ps[i]).to(device))

    for model in local_models:
        model.train()

    return local_models  

def fjord_global_model_assignment(model_name, device, num_classes = 10):
    if model_name == 'resnet18':
        ps = [0.1,0.4,0.5,1.0]
        glob_model = resnet18_f(num_classes, ps).to(device)
    elif model_name == 'resnet34':
        ps = [0.1,0.4,0.5,1.0]
        glob_model = resnet34_f(num_classes, ps).to(device)
    elif model_name == 'resnet50':
        ps = [0.1,0.4,0.5,1.0]
        glob_model = resnet50_f(num_classes, ps).to(device)
    elif model_name == 'resnet56':
        ps = [0.1,0.4,0.5,1.0]
        glob_model = resnet56_f(num_classes, ps).to(device)
    elif model_name == 'resnet101':
        ps = [0.1,0.4,0.5,1.0]
        glob_model = resnet101_f(num_classes, ps).to(device)
    elif model_name == 'resnet110':
        ps = [0.1,0.4,0.5,1.0]
        glob_model = resnet110_f(num_classes, ps).to(device)
    glob_model.train()

    return glob_model

#==========================================================================
'''DetphFL 사용'''

def depth_model_assignment(model_name, num_classes, device):
    local_models = []
    auxiliary_models = []
    cut_point = [1,2,3,4]
    if model_name == 'resnet18':
        for i in range(len(cut_point)):
                local_models.append(resnet18_depthFL(num_classes, cut_point[i]))
    elif model_name == 'resnet34':
        for i in range(len(cut_point)):
                local_models.append(resnet34_depthFL(num_classes, cut_point[i]))
    elif model_name == 'resnet50':
        for i in range(len(cut_point)):
                local_models.append(resnet50_depthFL(num_classes, cut_point[i]))
    elif model_name == 'resnet56':
        for i in range(len(cut_point)):
                local_models.append(resnet56_depthFL(num_classes, cut_point[i]))
    elif model_name == 'resnet101':
        for i in range(len(cut_point)):
                local_models.append(resnet101_depthFL(num_classes, cut_point[i]))
    elif model_name == 'resnet110':
        for i in range(len(cut_point)):
                local_models.append(resnet110_depthFL(num_classes, cut_point[i]))
    
    '''auxiliary network'''
    for i in range(len(local_models)-1):
        # client-side model
        local_models[i].to(device)   
        local_models[i].train()

        # auxiliary network 
        input = torch.zeros((1,3,32,32)).to(device)  # input image에 따라 달라짐 
        output = local_models[i](input)
        ax_net = Aux_net(output[-1].shape[1], num_classes).to(device)               # <------------------ 종류바꿈
        ax_net.train()
        auxiliary_models.append(ax_net)   

    return copy.deepcopy(local_models),  copy.deepcopy(auxiliary_models), cut_point

def global_depth_model_assignment(cut_point, model_name, device, num_classes = 10):
    if model_name == 'resnet18':
        net_glob = resnet18_depthFL(num_classes, cut_point).to(device)
    elif model_name == 'resnet34':
        net_glob = resnet34_depthFL(num_classes, cut_point).to(device)
    elif model_name == 'resnet50':
        net_glob = resnet50_depthFL(num_classes, cut_point).to(device)
    elif model_name == 'resnet56':
        net_glob = resnet56_depthFL(num_classes, cut_point).to(device)
    elif model_name == 'resnet101':
        net_glob = resnet101_depthFL(num_classes, cut_point).to(device)
    elif model_name == 'resnet110':
        net_glob = resnet110_depthFL(num_classes, cut_point).to(device)
    net_glob.train()

    return net_glob    