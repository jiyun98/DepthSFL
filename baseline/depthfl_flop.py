from train.extract_weight import extract_submodel_weight_from_global_fjord, extract_submodel_weight_from_global
from train.train_fl import LocalUpdate_d, test_img_d
from train.avg import *
from train.model_assign_flop import *
from utils.options import args_parser_main
from utils.utils import  seed_everything
from data.dataset import load_data

import numpy as np
import time
import wandb
import copy
import random


def main_depthfl(args):
    # Load dataset
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    local_models, auxiliary_models, args.cut_point = depth_model_assignment(
        args.model_v, args.model_name, args.num_classes, args.device)
    args.num_models = len(args.cut_point)
    
    net_glob = copy.deepcopy(local_models[-1])
    net_glob.to(args.device)
    w_glob = net_glob.state_dict()
    
    acc_test_total = []

    program = args.name
    print(program)

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
        w_locals_a = []
        for i in range(args.num_models):
            w_locals_a.append([])

        m = max(int(args.frac * args.num_users), 1)     # 현재 round에서 학습에 참여할 클라이언트 수
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 전체 클라이언트 중 랜덤으로 학습할 참여 클라이언트 선택
        mlist = [_ for _ in range(args.num_models)]


        for idx in idxs_users:
            if args.mode == 'worst':    # worst mode의 의미. 
                dev_spec_idx = 0    
                model_idx = 0       
            else:
                dev_spec_idx = min(idx//(args.num_users//args.num_models-1), args.num_models-2) # 균등하게 모델이 분포되어 있는 것 같음 0 ~ args.num_models
                model_idx = dev_spec_idx # random.choice(mlist[0:dev_spec_idx+1])


            ax_model_select = auxiliary_models[:min(model_idx+1, args.num_models-1)]

            weight = extract_submodel_weight_from_global(
                glob_net = copy.deepcopy(net_glob),net = copy.deepcopy(local_models[model_idx]), model_i = model_idx)
            model_select = local_models[model_idx]
            model_select.load_state_dict(copy.deepcopy(weight))
        

            local = LocalUpdate_d(args, dataset = dataset_train, idxs = dict_users[idx], model_idx = model_idx)
            weight, weight_a, loss, acc = local.train(net= copy.deepcopy(model_select), net_ax = copy.deepcopy(ax_model_select)) # weight_a는 리스트

            w_locals.append([copy.deepcopy(weight), model_idx])
            for i in range(len(weight_a)):
                w_locals_a[i].append(weight_a[i])
            print('[Epoch : {}][User {} with cut point {}] [Loss  {:.3f} | Acc {:.3f}]'
                  .format(iter, idx, model_idx, loss, acc))
            wandb.log({"[Train] User {} loss".format(args.cut_point[model_idx]): loss,"[Train] User {} acc".format(args.cut_point[model_idx]): acc}, step = iter)

            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))

        w_glob = HeteroAvg(w_locals)
        auxiliary_models = HeteroAvg_auxnet(w_locals_a, auxiliary_models)

        net_glob.load_state_dict(w_glob)

        # evaluate
        if iter % 10 == 0:
            if args.mode == 'worst':
                ti = 1
            else:
                ti = args.num_models
            
            test_acc_list = []

            for ind in range(ti):
                if ind != ti-1:
                    model_e = copy.deepcopy(local_models[ind])
                    model_a = copy.deepcopy(auxiliary_models[ind])

                    weight = extract_submodel_weight_from_global(glob_net = copy.deepcopy(net_glob),net = copy.deepcopy(model_e), model_i = model_idx)
                    model_e.load_state_dict(weight)

                    acc_test , loss_test = test_img_d(model_e, dataset_test, args, model_a)
                    print("[Epoch {}]Testing accuracy with cut point {} :  {:.2f}  ".format(iter, ind+1,  acc_test))
                    wandb.log({"[Test] User {} loss".format(args.cut_point[ind]): loss_test, "[Test] User {} acc".format(args.cut_point[ind]): acc_test}, step = iter)
                    test_acc_list.append(acc_test)
                else:
                    model_e = copy.deepcopy(local_models[ind])

                    weight = extract_submodel_weight_from_global(glob_net = copy.deepcopy(net_glob),net = copy.deepcopy(model_e), model_i = model_idx)
                    model_e.load_state_dict(weight)

                    acc_test , loss_test = test_img_d(model_e,  dataset_test, args)
                    print("[Epoch {}]Testing accuracy with cut point {} :  {:.2f} ".format(iter, ind+1,  acc_test))
                    wandb.log({"[Test] User {} loss".format(args.cut_point[ind]): loss_test, "[Test] User {} acc".format(args.cut_point[ind]): acc_test}, step = iter)

                    test_acc_list.append(acc_test)
            acc_test_total.append(test_acc_list)
        if iter % 50 == 0:
            print(program)
    print("finish")


    # Save output data to .excel file
    acc_test_arr = np.array(acc_test_total)
    file_name = './output2/DEP_KD/' + args.name + '/test_accuracy.txt'
    np.savetxt(file_name, acc_test_arr)

