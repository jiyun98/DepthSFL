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


if __name__ == '__main__':
    start_time = time.time()
    
    # Argument setting
    args = args_parser_main()
    args.device = 'cuda:' + args.device_id
    seed_everything(args.seed)
    # Load dataset
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    # Split point setting
    args.cut_point = 1

    # wandb setting
    wandb.init(project = '[New]Baseline')
    wandb.run.name = args.run_name
    wandb.config.update(args)

    net_glob = global_model_assignment(args.cut_point, args.model_name, args.device)
    w_glob = net_glob.state_dict()

    lr = args.lr

    acc_test_total = []

    # program = '{}_{}_on_{}_with_{}_users_{}_epochs_seed_{}_ps_{}.txt'.format('SFL',args.model_name, args.data, args.num_users, args.epochs, args.seed, args.ps)
    program = '{}_{}_on_{}_with_{}_users_split_point_{}_epochs_{}_seed_{}'.format(
        'FL_reduction',args.model_name, args.data, args.num_users, args.cut_point, args.epochs, args.seed)

    print(program)

    for iter in range(1, args.epochs+1):
        if iter == args.epochs/2:
            lr = lr*0.1
        elif iter == 3*args.epochs/4:
            lr = lr*0.1

        loss_locals = []
        acc_locals = []

        w_locals = []
        w_locals.append([w_glob, args.selected_idx])

        m = max(int(args.frac * args.num_users), 1)     # 현재 round에서 학습에 참여할 클라이언트 수
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 전체 클라이언트 중 랜덤으로 학습할 참여 클라이언트 선택

        for idx in idxs_users:
            local = LocalUpdate(args, dataset = dataset_train, idxs = dict_users[idx])
            w, loss, acc = local.train(net = copy.deepcopy(net_glob).to(args.device))

            w_locals.append([copy.deepcopy(w),args.cut_point-1])

            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))
            
            print('[Epoch : {}][User {} with split point {}] [Loss  {:.3f} | Acc {:.3f}]'
                  .format(iter, idx, args.cut_point, loss, acc))
            wandb.log({"[Train] Client {} loss".format(args.cut_point): loss,"[Train] Client {} acc".format(args.cut_point): acc}, step = iter)

        w_glob = HeteroAvg(w_locals)
        net_glob.load_state_dict(w_glob)


        if iter % 10 == 0 :
            test_acc, test_loss = test_img(net_glob, dataset_test, args)
            acc_test_total.append(test_acc)
            wandb.log({"[Test] loss": test_loss,"[Test] acc": test_acc}, step = iter)
#     
    print("finish")

    acc_test_arr = np.array(acc_test_total)
    file_name = './results/{}/{}_{}_on_{}_with_{}_users_split_point_{}_epochs_{}_seed_{}.txt'.format(
        args.model_name,'FL_reduction',args.model_name, args.data, args.num_users, args.cut_point, args.p, args.epochs, args.seed)

    np.savetxt(file_name, acc_test_arr)

    
    # df_c = pd.DataFrame(acc_test_arr_c)
    # df_s = pd.DataFrame(acc_test_arr_s)

    # with pd.ExcelWriter(file_name_excel) as writer:  
    #     df_c.to_excel(file_name_excel, index = False, sheet_name = 'client')
    #     df_s.to_excel(file_name_excel, index = False, isheet_name = 'server')
    print("Finish! 소요 시간은 ", time.time() - start_time, " 입니다.")


