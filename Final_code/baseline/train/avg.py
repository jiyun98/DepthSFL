import copy
import torch
import math
# from math import ceil as up

def HeteroAvg(w): # [0]: weight, [1]: dropout probability
    w_glob = copy.deepcopy(w[0][0])
    w_avg = copy.deepcopy(w[0][0])

    '''initialization'''
    for key in w_avg.keys():
        w_avg[key] = 0*w_avg[key]

    for key in w_glob.keys():
        cnt = 0
        for i in range(1, len(w)):
            w_local = copy.deepcopy(w[i][0])
            model_idx = w[i][1]

            if key in w_local.keys():
                cnt += 1
                w_avg[key] += w_local[key] 
        if cnt == 0:
            w_avg[key] = w_glob[key]
        else:
            w_avg[key] = w_avg[key] / cnt
    return w_avg

def HeteroAvg_auxnet(w, auxiliary_models): 
    # w는 [[aux_net 1 weight 모음],[aux_net 2 weight 모음],[aux_net 3 weight 모음],[aux_net 4 weight 모음],[aux_net 5 weight 모음]]으로 이뤄진 list임
    aux_models = copy.deepcopy(auxiliary_models)
    for idx in range(len(w)):
        if not w[idx]:    # w[i]가 비어있음
            continue
        else:
            num = len(w[idx])   # w[idx]를 사용한 클라이언트 수
            w_avg = copy.deepcopy(w[idx][0])
            # Initialization
            for key in w_avg.keys():
                w_avg[key] = 0*w_avg[key]
            # Aggregation
            for key in w_avg.keys():
                for i in range(num):
                    w_local = copy.deepcopy(w[idx][i])
                    w_avg[key] += w_local[key]
                w_avg[key] = w_avg[key] / num
            aux_models[idx].load_state_dict(w_avg)
    return aux_models

def HeteroAvg_cs(w_c, w_s): # [0]: weight, [1]: dropout probability
    w_c_glob = copy.deepcopy(w_c[0][0])
    
    w_c_avg = copy.deepcopy(w_c[0][0])
    # w_s_avg = copy.deepcopy(w_s[0][0])

    '''initialization'''
    for key in w_c_avg.keys():
        w_c_avg[key] = 0*w_c_avg[key]

    for key in w_c_glob.keys():
        cnt = 0
        for i in range(1, len(w_c)):    # len(w_c) = 한 라운드에 참여한 클라이언트 수 + 1
            w_c_local = copy.deepcopy(w_c[i][0])
            w_s_local = copy.deepcopy(w_s[i][0])
            if key in w_c_local.keys():
                cnt += 1
                w_c_avg[key] += w_c_local[key] 
            elif key in w_s_local.keys():
                cnt += 1
                w_c_avg[key] += w_s_local[key] 
        if cnt == 0:
            w_c_avg[key] = w_c_glob[key]
        else:
            w_c_avg[key] = w_c_avg[key] / cnt
    return w_c_avg


def HeteroAvg_new(w, BN_layers, com_layers, sing_layers,bn_keys, args, cs_opt): # [0]: weight, [1]: dropout probability
    w_glob = copy.deepcopy(w[0][0])
    w_avg = copy.deepcopy(w[0][0])

    b = [0] * args.num_models
    for i in range(1, len(w)):
        ind = w[i][1]
        b[ind] += 1
    '''initialization'''
    # 1) w/o bn layer 제외
    for key in w_avg.keys():
        if 'bn' not in key:
            w_avg[key] = 0*w_avg[key]
    # 2) bn layer
    for k in range(len(BN_layers)):
        if b[k]!=0:
            for key in BN_layers[k].keys():
                if 'num_batches_tracked' not in key:
                    BN_layers[k][key] = 0*BN_layers[k][key]

    for i in range(1, len(w)):
        index = w[i][1]
        for key in BN_layers[index].keys():
            if 'num_batches_tracked' not in key:
                BN_layers[index][key] += w[i][0][key]/b[index]
            else:
                BN_layers[index][key] += w[i][0][key]
                
    '''Averaging'''
    for key in w_avg.keys():
        if 'bn' in key or 'downsample.1' in key:
            pass
        cnt = 0
        for i in range(1, len(w)):
            w_local = copy.deepcopy(w[i][0])
            if key in w_local.keys():
                cnt +=1
                w_avg[key] += w_local[key]
        if cnt!=0:
            w_avg[key] = torch.div(w_avg[key], cnt)
        else:
            w_avg[key] = w_glob[key]
    if cs_opt == 'c':
        for key in BN_layers[-1].keys():
            w_avg[key] = copy.deepcopy(BN_layers[-1][key])
    elif cs_opt == 's':    
        for key in BN_layers[0].keys():
            w_avg[key] = copy.deepcopy(BN_layers[0][key])
    return w_avg, BN_layers

def F_Avg(w, bn_keys, args): # [0]: weight, [1]: dropout probability

    w_avg = copy.deepcopy(w[0][0]) # global model
    
    a = [0]*(len(w)-1) # 10
    acc_num = [len(w)-1]*(args.num_models+1) # [10]*5+1. accumulated number of models.
    b = [0]*args.num_models # 5. number of model idxs for federation.

    for n in range(1, len(w)):
        a[n-1] = w[n][1]
        
    for i in range(len(b)):
        b[i] = a.count(i)
        acc_num[i+1] = acc_num[i] - b[i]

    zero_idx = 0
    for o in range(1,args.num_models):
        if acc_num[o] == 0:
            break
        zero_idx += 1
    max_p_updated = args.ps[zero_idx] # define a maximum size of model that is updated
    # print(max_p_updated)
    '''
      Initialization
    '''
    for key in w_avg.keys():
        shape = w_avg[key].shape
        if len(shape) == 4:
            if key == 'conv1.weight':
                w_avg[key][0:up(shape[0]*max_p_updated), :, :, :] = 0*w_avg[key][0:up(shape[0]*max_p_updated), :, :, :]
            else: # other conv layers (layer1.0.conv1.weight -> shape = [16, 16, 3, 3]. layer2.0.conv1.weight -> hsape = [32, 16, 3, 3])
                w_avg[key][0:up(shape[0]*max_p_updated), 0:up(shape[1]*max_p_updated), :, :] = 0*w_avg[key][0:up(shape[0]*max_p_updated), 0:up(shape[1]*max_p_updated), :, :]

        elif len(shape) == 2: # linear.weight, 
            w_avg[key][:, 0:up(shape[1]*max_p_updated)] = 0*w_avg[key][:, 0:up(shape[1]*max_p_updated)]
        
        else: # bn1.weight
            w_avg[key] = 0*w_avg[key]   
    list_ps = copy.deepcopy(args.ps)
    list_ps.insert(0, 0)
    '''Aggregation'''
    for key in w_avg.keys():
        if key not in bn_keys:
            shape = w_avg[key].shape
            for i in range(1, len(w)):
                index = w[i][1]
                for j in range(index+1):
                    if len(shape) > 1:  # conv1.weight & linear.weight
                        li = up(shape[0]*list_ps[j])
                        hi = up(shape[0]*list_ps[j+1])
                        li2 = up(shape[1]*list_ps[j])
                        hi2 = up(shape[1]*list_ps[j+1])
                        if len(shape) == 4:
                            if key == 'conv1.weight': # w[key][out_ch, in_ch, kernel_size, kernel_size]
                                w_avg[key][li:hi, :, :, :] += w[i][0][key][li:hi, :, :, :]/acc_num[j]
                            else: # other conv layers
                                w_avg[key][li:hi, li2:hi2, :, :] += w[i][0][key][li:hi, li2:hi2, :, :]/acc_num[j]
                                if j!=0:
                                    w_avg[key][0:li, li2:hi2, :, :] += w[i][0][key][0:li, li2:hi2, :, :]/acc_num[j]
                                    w_avg[key][li:hi, 0:li2, :, :] += w[i][0][key][li:hi, 0:li2, :, :]/acc_num[j]
                        elif len(shape) == 2: # linear.weight
                            w_avg[key][:, li2:hi2] += w[i][0][key][:, li2:hi2]/acc_num[j]
                if key == 'linear.bias': # independent of j
                    w_avg[key] += w[i][0][key]/acc_num[0]
        else:
            shape = w_avg[key].shape
            for i in range(1, len(w)):
                index = w[i][1]
                for j in range(index+1):
                    if len(shape) > 0:
                        li = up(shape[0]*list_ps[j])
                        hi = up(shape[0]*list_ps[j+1])
                        w_avg[key][li:hi] += w[i][0][key][li:hi]/acc_num[j]
                    else: 
                        w_avg[key] = w_avg[key] +  w[i][0][key]/acc_num[j]
    return w_avg

def up(value):
  return math.ceil(value)