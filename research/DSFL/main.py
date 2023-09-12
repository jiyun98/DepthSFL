'''baseline을 실행하는 코드'''

import numpy as np
import time
import wandb
import copy
import os

from utils.options import args_parser_main
from utils.utils import *
from main_DepthSFL import *
from main_DDepthSFL import *
 

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser_main()
    args.wandb = True
    # Argument setting
    args.device = 'cuda:' + args.device_id
    seed_everything(args.seed)

    
    # filename setting
    args.name, wandb_name, timestamp = set_filename(args)
    filename = './output/' + args.name

    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run  = wandb.init(dir = filename, project = 'Proposed',
                          name = str(wandb_name), reinit = True,
                          settings = wandb.Settings(code_dir = "."))
        wandb.config.update(args)
    
    # logger = get_logger(logpath = os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))
    
    '''method 종류로 구분'''
    if args.method == 'dsfl':  
        main_DepthSFL(args)
        with open('./logs/learning_log.txt', 'a+') as f:     #  selected idx로 global model size 설정
            f.write("[Method:{}], [Data:{}], [Model:{}], [Cutpoint:{}], [Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n".format(
                        args.method, args.data, args.model_name, args.cut_point, 
                        args.seed, args.epochs, args.bs, args.frac, timestamp))
   
    elif args.method == 'ddsfl':    
        main_DDepthSFL(args)
        if args.kd_self_opt:
            args.method = 'sddsfl'
            with open('./logs/learning_log.txt', 'a+') as f: # selected idx로 global model size 설정
                f.write("[Method:{}], [Data:{}], [Model:{}], [Cutpoint:{}], [Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n".format(
                            args.method, args.data, args.model_name, args.cut_point, 
                            args.seed, args.epochs, args.bs, args.frac, timestamp))
        elif args.kd_server_opt:
            args.method = 'kddsfl'
            with open('./logs/learning_log.txt', 'a+') as f: # selected idx로 global model size 설정
                f.write("[Method:{}], [Data:{}], [Model:{}], [Cutpoint:{}], [Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n".format(
                            args.method, args.data, args.model_name, args.cut_point, 
                            args.seed, args.epochs, args.bs, args.frac, timestamp))
        else:
            with open('./logs/learning_log.txt', 'a+') as f: # selected idx로 global model size 설정
                f.write("[Method:{}], [Data:{}], [Model:{}], [Cutpoint:{}], [Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n".format(
                            args.method, args.data, args.model_name, args.cut_point, 
                            args.seed, args.epochs, args.bs, args.frac, timestamp))