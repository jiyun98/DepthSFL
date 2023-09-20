# Vanilla + DepthFL
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd

import time
import copy
import random
import wandb
# from torch.sutils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


from train.avg import HeteroAvg, HeteroAvg_auxnet, HeteroAvg_cs
from train.train_DDepthSFL_new3 import *
from train.extract_weight import extract_submodel_weight_from_global
from train.DDepthSFL_model_assign_flop import *
from data.dataset import load_data


def main_DDepthSFL(args):
    # Load dataset
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    # local model definition
    local_cmodels, local_smodels, auxiliary_models, args.cut_point = model_assignment(args.model_v, args.model_name, args.num_classes, args.device)
    args.num_models = len(args.cut_point)
    
    # Global model definition
    # net_glob_client, net_glob_server = global_model_assignment(args.cut_point, args.model_name, args.device, args.num_classes)
    net_glob_client = copy.deepcopy(local_cmodels[-1])
    net_glob_server = copy.deepcopy(local_smodels[0])

    acc_test_total_c = []
    acc_test_total_s = []

    program = args.name
    print(program)
        
    for iter in range(1,args.epochs+1):
        # learning rate update
        if args.learnable_step:
            if iter == args.epochs/2:
                args.lr = args.lr*0.1
                args.lr_server = args.lr_server * 0.1
            elif iter == 3*args.epochs/4:
                args.lr = args.lr*0.1
                args.lr_server = args.lr_server * 0.1
        if iter < args.kd_epoch:
            args.kd_gkt_opt = False
        else:
            args.kd_gkt_opt = True 
        w_glob_client = net_glob_client.state_dict()
        w_glob_server = net_glob_server.state_dict()

        w_locals_c = []
        w_locals_c.append([w_glob_client, args.num_models-1])    
        w_locals_s = []
        w_locals_s.append([w_glob_server, args.num_models-1])
        w_locals_a = []
        for i in range(args.num_models):
            w_locals_a.append([])

        m = max(int(args.frac * args.num_users), 1)     # 현재 round에서 학습에 참여할 클라이언트 수
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 전체 클라이언트 중 랜덤으로 학습할 참여 클라이언트 선택

        for idx in idxs_users:
            if args.mode == 'worst':  
                dev_spec_idx = 0    
                model_idx = 0       
            else:
                dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
                model_idx =  dev_spec_idx
            
            # Extract local client-side model from global client-side model
            weight_c = extract_submodel_weight_from_global(
                copy.deepcopy(net_glob_client), local_cmodels[model_idx], model_i = model_idx)
            c_model_select = local_cmodels[model_idx]
            c_model_select.load_state_dict(copy.deepcopy(weight_c))
            
            # Extract local server-side model from global server-side model
            weight_s = extract_submodel_weight_from_global(
                copy.deepcopy(net_glob_server), local_smodels[model_idx], model_i = model_idx)
            s_model_select = local_smodels[model_idx]
            s_model_select.load_state_dict(weight_s)

            aux_client = copy.deepcopy(auxiliary_models[:model_idx+1])
            aux_server = copy.deepcopy(auxiliary_models[model_idx:])

            # Train
            local_c = LocalUpdate_client(args,  dataset = dataset_train, idxs = dict_users[idx], wandb = wandb, model_idx = model_idx)
            smashed_data, label = local_c.get_smashed_data(net = copy.deepcopy(c_model_select))
            local_s = LocalUpdate_server(args, smashed_data = smashed_data, 
                                         label = label, wandb = wandb, model_idx = model_idx)
            kd_logits_server = local_s.get_kd_logits(net = copy.deepcopy(s_model_select))
            weight_c, weight_a_c, kd_logits_client,  args, loss_c, acc_c = local_c.train(
                    net_client = copy.deepcopy(c_model_select), net_ax = copy.deepcopy(aux_client), kd_logits = kd_logits_server)
            weight_s, weight_a_s, args, loss_s, acc_s = local_s.train(
                net = copy.deepcopy(s_model_select), net_ax = copy.deepcopy(aux_server), kd_logits = kd_logits_client)
            
            w_locals_c.append([copy.deepcopy(weight_c), model_idx])
            w_locals_s.append([copy.deepcopy(weight_s), model_idx])

            for i in range(args.num_models):
                if i < model_idx: # [0,1,2,3] 가정 model_idx = 1 -> [0,1] [1,2,3]
                    w_locals_a[i].append(weight_a_c[i])
                elif i == model_idx:
                    w_locals_a[i].append(weight_a_c[i])
                    w_locals_a[i].append(weight_a_s[i-model_idx])
                elif i > model_idx : 
                    w_locals_a[i].append(weight_a_s[i-model_idx])
            
            print('[Epoch : {}][User {} with cut_point {}] [C_Loss  {:.3f} | C_Acc {:.3f}] [S_Loss  {:.3f} | S_Acc {:.3f}]'
                  .format(iter, idx, args.cut_point[model_idx], loss_c, acc_c,loss_s, acc_s))
            wandb.log({"[Train] Client {} loss".format(args.cut_point[model_idx]): loss_c,"[Train] Client {} acc".format(args.cut_point[model_idx]): acc_c, \
                        "[Train] Server {} loss".format(args.cut_point[model_idx]): loss_s,"[Train] Server {} acc".format(args.cut_point[model_idx]): acc_s}, step = iter)


        # 서버 쪽 글로벌 모델의 일부를 fed server로 보낸다고 가정합시다.
        w_c_glob = HeteroAvg(w_locals_c) #  -> w_locals_s를 다 보내지 말고 중복되는 부분만 보내면 될 듯
        w_s_glob = HeteroAvg(w_locals_s)
        auxiliary_models = HeteroAvg_auxnet(w_locals_a, auxiliary_models)

        net_glob_client.load_state_dict(w_c_glob)
        net_glob_server.load_state_dict(w_s_glob)

        # Evaluate
        if iter % 10 == 0:
            if args.mode == 'worst':
                ti = 1
            else:
                ti = args.num_models
            
            test_acc_list_c = []
            test_acc_list_s = []

            for ind in range(ti):
                model_e_c = copy.deepcopy(local_cmodels[ind])
                model_e_s = copy.deepcopy(local_smodels[ind])
                model_e_a = copy.deepcopy(auxiliary_models[ind])

                weight_c = extract_submodel_weight_from_global(
                    copy.deepcopy(net_glob_client), model_e_c, model_i = ind)
                weight_s = extract_submodel_weight_from_global(
                    copy.deepcopy(net_glob_server), model_e_s, model_i = ind)
                                
                model_e_c.load_state_dict(weight_c)
                model_e_s.load_state_dict(weight_s)

                acc_test_c, loss_test_c, acc_test_s, loss_test_s = test_img(model_e_c, model_e_s, model_e_a, dataset_test, args)
                print("[Epoch {}]Testing accuracy with split point {} : [Client : {:.2f} | Server : {:.2f}] ".format(iter,args.cut_point[ind], acc_test_c, acc_test_s))
                
                wandb.log({"[Test] Client {} acc".format(ind+1): acc_test_c, "[Test] Server {} acc".format(ind+1): acc_test_s}, step = iter)

                test_acc_list_c.append(acc_test_c)
                test_acc_list_s.append(acc_test_s)
            acc_test_total_c.append(test_acc_list_c)
            acc_test_total_s.append(test_acc_list_s)
        if iter % 50 == 0:
            print(program)
    print("finish")


    # Save output data to .excel file
    acc_test_arr_c = np.array(acc_test_total_c)
    acc_test_arr_s = np.array(acc_test_total_s)
    file_name_c = './output_gkt/{}/'.format(args.method_name) + args.name + '/[client]test_accuracy.txt'
    file_name_s = './output_gkt/{}/'.format(args.method_name) + args.name + '/[server]test_accuracy.txt'

    np.savetxt(file_name_c, acc_test_arr_c)
    np.savetxt(file_name_s, acc_test_arr_s)
# 
#     # Save the final trained model
#     torch.save(net_glob_client.state_dict(), "./saved/saved_model/{}/client/client_{}_{}_on_{}_with_{}_users_split_point_{}_epochs_{}_seed_{}.pth".format(
#         args.model_name,'D_DepthSFL', args.model_name, args.data, args.num_users, args.cut_point,args.ps, args.epochs, args.seed))
#     torch.save(net_glob_server.state_dict(), "./saved/saved_model/{}/server/server_{}_{}_on_{}_with_{}_users_split_point_{}_epochs_{}_seed_{}.pth".format(
#         args.model_name,'D_DepthSFL', args.model_name, args.data, args.num_users, args.cut_point,args.ps, args.epochs, args.seed))
#     for i in range(args.num_models):
#         torch.save(net_glob_server.state_dict(), "./saved/saved_model/{}/auxnet/{}_{}_auxnet_{}_{}_on_{}_with_{}_users_split_point_{}_epochs_{}_seed_{}.pth".format(
#             args.model_name,'D_DepthSFL', args.fr, args.cut_point[i], args.model_name, args.data, args.num_users, args.cut_point,args.ps, args.epochs, args.seed))
# 
# 
