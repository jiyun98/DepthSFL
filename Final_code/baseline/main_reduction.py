from train.extract_weight import extract_submodel_weight_from_global
from train.train_fl import LocalUpdate, test_img
from train.avg import HeteroAvg
from train.model_assign import *
from utils.options import args_parser_main
from utils.utils import  seed_everything
from data.dataset import load_data

import numpy as np
import time
import wandb
import copy


def main_reduction(args):
    # Load dataset
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    net_glob, args.cut_point = FL_model_assignment(args.selected_idx, args.model_name, args.device, args.num_classes)
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

        m = max(int(args.frac * args.num_users), 1)     # 현재 round에서 학습에 참여할 클라이언트 수
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 전체 클라이언트 중 랜덤으로 학습할 참여 클라이언트 선택

        for idx in idxs_users:
            local = LocalUpdate(args, dataset = dataset_train, idxs = dict_users[idx])
            w, loss, acc = local.train(net = copy.deepcopy(net_glob).to(args.device))

            w_locals.append([copy.deepcopy(w),args.selected_idx-1])

            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))
            
            print('[Epoch : {}][User {} with split point {}] [Loss  {:.3f} | Acc {:.3f}]'
                  .format(iter, idx, args.selected_idx, loss, acc))
            wandb.log({"[Train] User {} loss".format(args.selected_idx): loss,"[Train] User {} acc".format(args.selected_idx): acc}, step = iter)

        w_glob = HeteroAvg(w_locals)
        net_glob.load_state_dict(w_glob)


        if iter % 10 == 0 :
            test_acc, test_loss = test_img(net_glob, dataset_test, args)
            acc_test_total.append(test_acc)
            print('[Test][User {} with split point {}] [Loss  {:.3f} | Acc {:.3f}]'
                  .format(idx, args.selected_idx, test_loss, test_acc))
            wandb.log({"[Test] User {} loss".format(args.selected_idx): test_loss,"[Test] User {} acc".format(args.selected_idx): test_acc}, step = iter)
#     
    print("finish")

    acc_test_arr = np.array(acc_test_total)
    file_name = './output/RED/' + args.name + '/test_accuracy.txt'
    np.savetxt(file_name, acc_test_arr)

