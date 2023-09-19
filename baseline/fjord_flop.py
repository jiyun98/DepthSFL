from train.extract_weight import extract_submodel_weight_from_global_fjord
from train.train_fl import LocalUpdate, test_img
from train.avg import F_Avg
from train.model_assign_flop import *
from utils.options import args_parser_main
from utils.utils import  seed_everything
from data.dataset import load_data

import numpy as np
import time
import wandb
import copy
import random


def main_fjord(args):
    # Load dataset
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    if args.model_name == 'resnet18' or args.model_name == 'resnet34':
        args.ps = [0.1, 0.53, 0.87, 1.0]
    elif args.model_name == 'resnet50':
        args.ps = [0.1, 0.25, 0.6, 1.0]
    elif args.model_name == 'resnet101':
        args.ps = [0.1, 0.2, 0.8, 1.0]
    elif args.model_name == 'resnet56' or args.model_name == 'resnet110':
        args.ps = [0.2,0.5, 1.0]
    args.num_models = len(args.ps)


    local_models = fjord_local_model_assignment(
        args.ps, args.model_name, args.device, args.num_classes)

    net_glob = fjord_global_model_assignment(
        args.ps[-1], args.model_name, args.device, args.num_classes)
    
    w_glob = net_glob.state_dict()
    
    bn_keys = []
    for i in w_glob.keys():
        if 'bn' in i or 'shortcut.1' in i:
            bn_keys.append(i)


    program = args.name
    print(program)
    
    acc_test_total = []
    for iter in range(1,args.epochs+1):
        # learning rate update
        if iter == args.epochs/2:
            args.lr = args.lr*0.1
        elif iter == 3*args.epochs/4:
            args.lr = args.lr*0.1

        loss_locals = []
        acc_locals = []

        w_glob = net_glob.state_dict()

        w_locals = []
        w_locals.append([w_glob, args.num_models-1])    

        m = max(int(args.frac * args.num_users), 1)     # 현재 round에서 학습에 참여할 클라이언트 수
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 전체 클라이언트 중 랜덤으로 학습할 참여 클라이언트 선택
        mlist = [_ for _ in range(args.num_models)]


        for idx in idxs_users:
            if args.mode == 'worst':    # worst mode의 의미. 
                dev_spec_idx = 0    
                model_idx = 0       
            else:
                dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-2) # 균등하게 모델이 분포되어 있는 것 같음 0 ~ args.num_models
                model_idx = random.choice(mlist[0:dev_spec_idx+1])

            p_select = args.ps[model_idx] # 해당하는 모델에 대한 p 값 얻어옴

            weight = extract_submodel_weight_from_global_fjord(net = copy.deepcopy(net_glob), p = p_select, model_i = model_idx)
            model_select = local_models[model_idx]
            model_select.load_state_dict(copy.deepcopy(weight))
        

            local = LocalUpdate(args, p_select, dataset = dataset_train, idxs = dict_users[idx])
            weight, loss, acc = local.train(net= copy.deepcopy(model_select))

            w_locals.append([copy.deepcopy(weight), model_idx])

            print('[Epoch : {}][User {} with dropout {}] [Loss  {:.3f} | Acc {:.3f}]'
                  .format(iter, idx, p_select, loss, acc))
            wandb.log({"[Train] User {} loss".format(model_idx+1): loss,"[Train] User {} acc".format(model_idx+1): acc}, step = iter)

            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))

        w_glob = F_Avg(w_locals, bn_keys, args)

        net_glob.load_state_dict(w_glob)

        # evaluate
        if iter % 10 == 0:
            if args.mode == 'worst':
                ti = 1
            else:
                ti = args.num_models
            
            test_acc_list = []

            for ind in range(ti):
                p = args.ps[ind]
                model_e = copy.deepcopy(local_models[ind])

                weight = extract_submodel_weight_from_global_fjord(net = copy.deepcopy(net_glob), p = p, model_i = ind)
                model_e.load_state_dict(weight)

                acc_test , loss_test = test_img(model_e, dataset_test, args)
                print("[Epoch {}]Testing accuracy with dropout {} :  {:.2f}  ".format(iter,p, acc_test))
                wandb.log({"[Test] User {} loss".format(ind+1): loss_test, "[Test] User {} acc".format(ind+1): acc_test}, step = iter)

                test_acc_list.append(acc_test)
            acc_test_total.append(test_acc_list)
        if iter % 50 == 0:
            print(program)
    print("finish")

 
    # Save output data to .excel file
    acc_test_arr = np.array(acc_test_total)
    file_name = './output2/FJO/' + args.name + '/test_accuracy.txt'
    np.savetxt(file_name, acc_test_arr)

    
