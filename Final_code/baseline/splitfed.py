from train.extract_weight import extract_submodel_weight_from_global_fjord, extract_submodel_weight_from_global
from train.train_fl import *
from train.avg import *
from train.model_assign import *
from utils.options import args_parser_main
from utils.utils import  seed_everything
from data.dataset import load_data

import numpy as np
import time
import wandb
import copy
import random


def main_sfl(args):
    # Load dataset
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    local_cmodels, local_smodels, args.cut_point = SFL_model_assignment(
        args.selected_idx, args.model_name, args.num_classes, args.device)
    args.num_models = len(args.cut_point)
    net_glob_client = copy.deepcopy(local_cmodels[-1])
    net_glob_server = copy.deepcopy(local_smodels[0])
    
    # Train
    acc_test_total_c = []
    acc_test_total_s = []

    program = args.name
    print(program)

    for iter in range(1,args.epochs+1):
        # learning rate update
        if args.learnable_step:
            if iter == args.epochs/2:
                args.lr = args.lr*0.1
            elif iter == 3*args.epochs/4:
                args.lr = args.lr*0.1
        w_glob_client = net_glob_client.state_dict()
        w_glob_server = net_glob_server.state_dict()

        w_locals_c = []
        w_locals_c.append([w_glob_client, args.num_models-1])    
        w_locals_s = []
        w_locals_s.append([w_glob_server, args.num_models-1])


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

            # Train
            local = LocalUpdate_sfl(args,  dataset = dataset_train, idxs = dict_users[idx], wandb = wandb, model_idx = model_idx)
            weight_c, weight_s,  args,loss_s, acc_s = local.train(
                    net_client = copy.deepcopy(c_model_select), net_server = copy.deepcopy(s_model_select))

            w_locals_c.append([copy.deepcopy(weight_c), model_idx])
            w_locals_s.append([copy.deepcopy(weight_s), model_idx])


            print('[Epoch : {}][User {} with cut_point {}] [S_Loss  {:.3f} | S_Acc {:.3f}]'
                  .format(iter, idx, args.cut_point[model_idx], loss_s, acc_s))
            wandb.log({"[Train] Server {} loss".format(args.cut_point[model_idx]): loss_s,"[Train] Server {} acc".format(args.cut_point[model_idx]): acc_s}, step = iter)


        # 서버 쪽 글로벌 모델의 일부를 fed server로 보낸다고 가정합시다.
        w_c_glob = HeteroAvg(w_locals_c) #  -> w_locals_s를 다 보내지 말고 중복되는 부분만 보내면 될 듯
        w_s_glob = HeteroAvg(w_locals_s)

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

                weight_c = extract_submodel_weight_from_global(
                    copy.deepcopy(net_glob_client), model_e_c, model_i = ind)
                weight_s = extract_submodel_weight_from_global(
                    copy.deepcopy(net_glob_server), model_e_s, model_i = ind)
                                
                model_e_c.load_state_dict(weight_c)
                model_e_s.load_state_dict(weight_s)

                acc_test_s, loss_test_s = test_img_sfl(model_e_c, model_e_s, dataset_test, args)
                print("[Epoch {}]Testing accuracy with split point {} :  {:.2f}] ".format(iter,args.cut_point[ind], acc_test_s))
                wandb.log({"[Test] Server {} acc".format(ind+1): acc_test_s}, step = iter)
                test_acc_list_s.append(acc_test_s)

            acc_test_total_s.append(test_acc_list_s)
        if iter % 50 == 0:
            print(program)
    print("finish")


    # Save output_v2 data to .excel file
    acc_test_arr_s = np.array(acc_test_total_s)

    file_name_s = './output/SFL/' + args.name + '/[server]test_accuracy_{}.txt'.format(args.seed)
   
    np.savetxt(file_name_s, acc_test_arr_s)
    
   