# multiple smashed data + depthfl
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from data.dataset import DatasetSplit
from utils.utils import calculate_accuracy
import copy
import torch.nn.functional as F

import numpy as np
import matplotlib

class LocalUpdate(object):
    def __init__(self, args, dataset = None, idxs = None, wandb = None, model_idx = None):
        '''
        args : argument
        dataset : 학습에 사용될 데이터 셋
        idxs : 데이터 분할에 사용되는
        '''
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.local_bs, shuffle = True)
        self.wandb = wandb
        self.model_idx = model_idx

    def train(self, net_client, net_ax,  net_server):
        net_client.to(self.args.device)
        net_server.to(self.args.device)

        net_client.train()
        net_server.train()
        # Auxiliary net 분리 (클라이언트 쪽/서버 쪽)
        aux_client = copy.deepcopy(net_ax[:self.model_idx+1])
        aux_server = copy.deepcopy(net_ax[self.model_idx + 1:])

        params = list(net_client.parameters())
        
        for ax in aux_client:
            ax.to(self.args.device)
            ax.train()
            params += list(ax.parameters())

        optimizer_client = torch.optim.SGD(params, lr = self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        criterion = nn.CrossEntropyLoss() 

        epoch_loss_c = []
        epoch_acc_c = []
        epoch_loss_s = []
        epoch_acc_s = []

        for iter in range(self.args.local_ep):
            batch_loss_c = []
            batch_acc_c = []
            batch_loss_s = []
            batch_acc_s = []
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net_client.zero_grad()
                optimizer_client.zero_grad()

                # forward prop in client-side model      
                fx = net_client(images) # [[output1,depth1],[output2,depth2],...]

                loss_client = 0
                
                for i in range(len(fx)):
                    auxiliary_net = aux_client[i]# fx[i][1]-1]
                    auxiliary_net.zero_grad()
                    client_logits, client_probs = auxiliary_net(fx[i])
                    loss_client += criterion(client_logits, labels)
                acc_client = calculate_accuracy(client_logits, labels)  

                # auxiliary_net = aux_client[-1]  
                # auxiliary_net.zero_grad()
                # client_logits, client_probs = auxiliary_net(fx[-1])
                # loss_client += criterion(client_logits, labels)
                # acc_client = calculate_accuracy(client_logits, labels) 

                batch_loss_c.append(loss_client.item())
                batch_acc_c.append(acc_client.item())    

                loss_client.backward()
                optimizer_client.step() 

                fx_client = fx[-1].clone().detach().requires_grad_(True)

                net_server , aux_server, self.args, loss_server, acc_server= train_server(fx_client, aux_server, labels, net_server, self.args)
                batch_loss_s.append(loss_server.item())
                batch_acc_s.append(acc_server.item())

            epoch_loss_c.append(sum(batch_loss_c)/len(batch_loss_c))
            epoch_acc_c.append(sum(batch_acc_c)/len(batch_acc_c))
            epoch_loss_s.append(sum(batch_loss_s)/len(batch_loss_s))
            epoch_acc_s.append(sum(batch_acc_s)/len(batch_acc_s))
        
        client_loss_acc = [sum(epoch_loss_c)/len(epoch_loss_c), epoch_acc_c[-1]]
        server_loss_acc = [sum(epoch_loss_s)/len(epoch_loss_s), epoch_acc_s[-1]]

        weight_a_c = [ax.state_dict() for ax in aux_client]
        weight_a_s = [ax.state_dict() for ax in aux_server]
        weight_a_c.extend(weight_a_s)
        # net_ax가 요소마다 데이트 되는지 확인해봐야 함
        return net_client.state_dict(), weight_a_c, net_server.state_dict(), self.args, client_loss_acc, server_loss_acc # sum(epoch_loss)/len(epoch_loss), epoch_acc[-1], sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]
       
def train_server(fx_client, aux_server, y, net, args, loss_opt = False): 
    net_server = copy.deepcopy(net)
    net_ax = copy.deepcopy(aux_server)

    params = list(net_server.parameters())
    
    for ax in net_ax:
        ax.to(args.device)
        ax.train()
        params += list(ax.parameters())

    optimizer_server = torch.optim.SGD(params, lr = args.lr, momentum = args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()  

    net_server.train()
    net_server.zero_grad()
    optimizer_server.zero_grad()

    fx_server,probs = net_server(fx_client) # fx_server가 list 형식으로!
    
    loss = 0
    for i in range(len(fx_server)-1):
        auxiliary_net = net_ax[i]# fx[i][1]-1]
        auxiliary_net.zero_grad()
        server_logits, server_probs = auxiliary_net(fx_server[i])
        loss += criterion(server_logits, y)
    loss += criterion(fx_server[-1], y)
    acc = calculate_accuracy(fx_server[-1], y)  

    loss.backward()
    optimizer_server.step()

    return net_server, copy.deepcopy(net_ax), args, loss, acc


def test_img(net_c, net_s, net_a, datatest, args):
    net_c.eval()
    net_c.to(args.device)

    net_s.eval()
    net_s.to(args.device)

    net_a.eval()
    net_a.to(args.device)

    # testing
    test_loss_c = 0
    test_loss_s = 0
    correct_c = 0
    correct_s = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader) # len(data_loader)= 469개, 128*468개 세트, 1개는 96개 들어있음

    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):
            if 'cuda' in args.device:
                data, target = data.to(args.device), target.to(args.device)
            fx_client = net_c(data)
            
            # client-side loss
            logits_c, probas_c = net_a(fx_client[-1])
            # server-side loss
            logits_s, probas_s = net_s(fx_client[-1])

            test_loss_c += F.cross_entropy(probas_c, target, reduction='sum').item()
            y_pred_c = probas_c.data.max(1, keepdim=True)[1]
            correct_c += y_pred_c.eq(target.data.view_as(y_pred_c)).long().cpu().sum()

            test_loss_s += F.cross_entropy(probas_s, target, reduction='sum').item()
            y_pred_s = probas_s.data.max(1, keepdim=True)[1]     # label return 
            correct_s += y_pred_s.eq(target.data.view_as(y_pred_s)).long().cpu().sum()  # ([32]) -> ([32, 1])

    test_loss_c /= len(data_loader.dataset)
    test_loss_s /= len(data_loader.dataset)
    accuracy_c = 100.00 * correct_c / len(data_loader.dataset)
    accuracy_s = 100.00 * correct_s / len(data_loader.dataset)

    return accuracy_c, test_loss_c, accuracy_s, test_loss_s

def plot_grad_flow_v2(named_parameters, wandb, names,idx):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().cpu().mean())
            max_grads.append(p.grad.abs().cpu().max())
    
    data1 = [[layer, ave_grads] for (layer, ave_grads) in zip(layers,ave_grads)]
    table1 = wandb.Table(data=data1, columns = ["layer", "average"])
    wandb.log({"gradient_flow(avg)_{}_{}".format(names,idx) : wandb.plot.bar(table1, "layer","average", title="gradient_flow(avg)_{}_{}".format(names,idx))})
    
    data = [[layer, max_grads] for (layer,  max_grads) in zip(layers,max_grads)]
    table = wandb.Table(data=data, columns = ["layer", "maximum"])
    wandb.log({"gradient_flow(max)_{}_{}".format(names,idx) : wandb.plot.bar(table, "layer", "maximum", title="gradient_flow(max)_{}_{}".format(names,idx))})
    return ave_grads, max_grads, layers






    