from train.extract_weight import extract_submodel_weight_from_global
from train.train_fl import LocalUpdate, test_img
from train.avg import HeteroAvg
from train.model_assign_flop import *
from utils.options import args_parser_main
from utils.utils import  seed_everything
from data.dataset import load_data

import numpy as np
import time
import wandb
import copy


def main_exclusive(args):

    # Load dataset
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)
    num_models = len(args.cut_point)

    net_glob = FL_model_assignment(args.selected_idx, args.model_name, args.device, args.num_classes)
    w_glob = net_glob.state_dict()

    lr = args.lr

    acc_test_total = []

    program = args.name
    print(program)

    for iter in range(1, args.epochs+1):
        if iter == args.epochs/2:
            args.lr = args.lr*0.1
        elif iter == 3*args.epochs/4:
            args.lr = args.lr*0.1

        loss_locals = []
        acc_locals = []

        w_locals = []
        w_locals.append([w_glob, args.selected_idx])
        
        # model_idx가 5인 애들만 학습에 참여하게 하자.
        m = max(int(args.frac * args.num_users), 1)     # 현재 round에서 학습에 참여할 클라이언트 수 -> 5
        numnum = args.num_users//num_models    # 10
        selected_list = list(range(numnum*(args.selected_idx-1),numnum*args.selected_idx))
        idxs_users = np.random.choice(selected_list, m, replace=False)  # 전체 클라이언트 중 랜덤으로 학습할 참여 클라이언트 선택
        model_idx = args.selected_idx - 1
        for idx in idxs_users:
            #dev_spec_idx = idx//(args.num_users//num_models) # 사실 굳이 필요 없음..나중에 선택한 글로벌 모델 크기가 4,3이 되면 그 때는 필요해지겠지
            local = LocalUpdate(args, dataset = dataset_train, idxs = dict_users[idx])
            w, loss, acc = local.train(net = copy.deepcopy(net_glob).to(args.device))

            w_locals.append([copy.deepcopy(w),model_idx])

            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))
            
            print('[Epoch : {}][User {} with split point {}] [Loss  {:.3f} | Acc {:.3f}]'
                  .format(iter, idx, model_idx, loss, acc)) # 
            wandb.log({"[Train] User {} loss".format(args.selected_idx): loss,"[Train] User {} acc".format(args.selected_idx): acc}, step = iter)

            
        w_glob = HeteroAvg(w_locals)
        net_glob.load_state_dict(w_glob)


        if iter % 10 == 0 :
            test_acc, test_loss = test_img(net_glob, dataset_test, args)
            acc_test_total.append(test_acc)
            wandb.log({"[Test] User {} loss".format(args.selected_idx): test_loss,"[Test] User {} acc".format(args.selected_idx): test_acc}, step = iter)
#     
    print("finish")

    acc_test_arr = np.array(acc_test_total)
    file_name = './output2/EXC/' + args.name + '/test_accuracy.txt'
    np.savetxt(file_name, acc_test_arr)

    
