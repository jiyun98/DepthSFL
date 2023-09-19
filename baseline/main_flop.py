'''baseline을 실행하는 코드'''

import numpy as np
import time
import wandb
import copy
import os

from utils.options import args_parser_main
from utils.utils import *
from main_exclusive_flop import *
from main_reduction_flop import *
from fjord_flop import *
from depthfl_flop import *


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser_main()
    args.kd_opt = True # depthfl 은 필수임!
    # Argument setting
    args.device = 'cuda:' + args.device_id
    seed_everything(args.seed)


    # filename setting
    args.name, wandb_name, timestamp, method_name = set_filename(args)
    filename = './output2/{}/'.format(method_name) + args.name

    if not os.path.exists(filename):
        os.makedirs(filename)

    if args.wandb:
        run  = wandb.init(dir = filename, project = 'BaseLine(0918)',
                          name = str(wandb_name), reinit = True,
                          settings = wandb.Settings(code_dir = "."))
        wandb.config.update(args)
    
    # logger = get_logger(logpath = os.path.join(filename, 'logs'), filepath=os.path.abspath(__file__))
    
    '''method 종류로 구분'''
    if args.method == 'exclusive':  # EXC
        main_exclusive(args)
        with open('./logs/learning_log.txt', 'a+') as f:     #  selected idx로 global model size 설정
            f.write("[Method:{}], [Data:{}], [Model:{}], [Cutpoint:{}], [Select idx:{}], [Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n ".format(
                        args.method, args.data, args.model_name, args.cut_point, args.selected_idx,
                        args.seed, args.epochs, args.bs, args.frac, timestamp))
   
    elif args.method == 'reduction':    # RED
        main_reduction(args)
        with open('./logs/learning_log.txt', 'a+') as f: # selected idx로 global model size 설정
            f.write("[Method:{}], [Data:{}], [Model:{}], [Cutpoint:{}], [Select idx:{}], [Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n".format(
                        args.method, args.data, args.model_name, args.cut_point, args.selected_idx,
                        args.seed, args.epochs, args.bs, args.frac, timestamp))
            
    elif args.method == 'fjord':    # FJO
        main_fjord(args)
        with open('./logs/learning_log.txt', 'a+') as f: # selected idx로 global model size 설정
            f.write("[Method:{}], [Data:{}], [Model:{}], [Ps:{}], [Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n".format(
                        args.method, args.data, args.model_name, args.ps, 
                        args.seed, args.epochs, args.bs, args.frac, timestamp))

    elif args.method == 'depthfl':  # DEP   # args.cut_point = [1,2,3,4]로 설정해주세요
        main_depthfl(args)
        with open('./logs/learning_log.txt', 'a+') as f: # selected idx로 global model size 설정
            f.write("[Method:{}], [Data:{}], [Model:{}], [Cutpoint:{}], [KD:{}],[Seed:{}], [Epoch:{}], [BatchSize:{}], [Frac:{}], [Time:{}]\n".format(
                        args.method, args.data, args.model_name, args.cut_point, args.kd_opt,
                        args.seed, args.epochs, args.bs, args.frac, timestamp))
     
    print("Complete!")
