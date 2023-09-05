
import argparse

def args_parser_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=50)
    parser.add_argument('--noniid', action='store_true') # default: false
    parser.add_argument('--class_per_each_client', type=int, default=10)

    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--bs', type=int, default= 64)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default= 1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--rs', type=int, default=0)

    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--device_id', type=str, default='1')
    parser.add_argument('--learnable_step', type=bool, default=True)

    parser.add_argument('--run_name',type=str,default = 'vanilla')

    parser.add_argument('--ps', type = float, default = 1.0)
    parser.add_argument('--cut_point', default = [])
    parser.add_argument('--seed', type = int, default = 123)

    # KD option
    parser.add_argument('--T', type = int, default = 4)
    parser.add_argument('--lambdaa', type = float, default = 0.05)
    parser.add_argument('--kd_epoch', type=int, default=50)
    parser.add_argument('--kd_opt', type=bool, default=False)
    
    args = parser.parse_args()

    return args