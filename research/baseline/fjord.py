from train.extract_weight import extract_submodel_weight_from_global_fjord
from train.train_fl import LocalUpdate, test_img
from train.avg import F_Avg
from train.model_assign import *
from utils.options import args_parser_main
from utils.utils import  seed_everything
from data.dataset import load_data

import numpy as np
import time
import wandb
import copy
import random


if __name__ == '__main__':
    start_time = time.time()
    
    # Argument setting
    args = args_parser_main()
    args.device = 'cuda:' + args.device_id
    seed_everything(args.seed)
    # Load dataset6
    dataset_train, dataset_test, dict_users, args.num_classes = load_data(args)

    # Split point setting
    args.ps = [0.25, 0.5, 1.0]
    args.num_models = len(args.ps)
    # wandb setting
    wandb.init(project = '[New]Baseline')
    wandb.run.name = args.run_name
    wandb.config.update(args)

    local_models = fjord_local_model_assignment(
        args.ps, args.model_name, args.device, args.num_classes)

    net_glob = fjord_global_model_assignment(
        args.ps[-1], args.model_name, args.device, args.num_classes)
    
    w_glob = net_glob.state_dict()
    
    bn_keys = []
    for i in w_glob.keys():
        if 'bn' in i or 'shortcut.1' in i:
            bn_keys.append(i)



    program = '{}_{}_on_{}_with_{}_users_{}_epochs_seed_{}_ps_{}.txt'.format(
        'FjORD',args.model_name, args.data, args.num_users, args.epochs, args.seed, args.ps)
    print(program)

    for iter in range(1,args.epochs+1):
        # learning rate update
        if iter == args.epochs/2:
            args.lr = args.lr*0.1
        elif iter == 3*args.epochs/4:
            args.lr = args.lr*0.1

        loss_locals = []
        acc_locals = []
        acc_test_total = []

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
                dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1) # 균등하게 모델이 분포되어 있는 것 같음 0 ~ args.num_models
                model_idx = random.choice(mlist[0:dev_spec_idx+1])

            p_select = args.ps[model_idx] # 해당하는 모델에 대한 p 값 얻어옴

            weight = extract_submodel_weight_from_global_fjord(net = copy.deepcopy(net_glob), p = p_select, model_i = model_idx)
            model_select = local_models[model_idx]
            model_select.load_state_dict(copy.deepcopy(weight))
        

            local = LocalUpdate(args, p_select, dataset = dataset_train, idxs = dict_users[idx])
            weight, loss, acc = local.train(net= copy.deepcopy(model_select))

            w_locals.append([copy.deepcopy(weight), model_idx])

            print('[Epoch : {}][User {} with dropout {}] [Loss  {:.3f} | Acc {:.3f}%]'
                  .format(iter, idx, p_select, loss, acc))
            

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
                print("[Epoch {}]Testing accuracy with dropout {} : [Client : {:.2f} | Server : {:.2f}] ".format(iter,p, acc_test, acc_test))

                test_acc_list.append(acc_test)
            acc_test_total.append(test_acc_list)
        if iter % 50 == 0:
            print(program)
    print("finish")


    # Save output data to .excel file
    acc_test_arr = np.array(acc_test_total)
    file_name = './results/{}/{}_{}_on_{}_with_{}_users_ps_{}_epochs_{}_seed_{}.txt'.format(
        args.model_name,'FjORD', args.model_name, args.data, args.num_users, args.ps, args.epochs, args.seed)
    
    np.savetxt(file_name, acc_test_arr)

    
    # df_c = pd.DataFrame(acc_test_arr_c)
    # df_s = pd.DataFrame(acc_test_arr_s)

    # with pd.ExcelWriter(file_name_excel) as writer:  
    #     df_c.to_excel(file_name_excel, index = False, sheet_name = 'client')
    #     df_s.to_excel(file_name_excel, index = False, sheet_name = 'server')
    print("Finish! 소요 시간은 ", time.time() - start_time, " 입니다.")

