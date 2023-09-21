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
# import matplotlib



class LocalUpdate_client(object):
    def __init__(self, args, dataset = None, idxs = None, wandb = None, model_idx = None, kd_gkt_opt = False):
        '''
        args : argument
        dataset : 학습에 사용될 데이터 셋
        idxs : 데이터 분할에 사용되는
        '''
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.local_bs, shuffle = True)
        self.wandb = wandb
        self.model_idx = model_idx
        self.kd_gkt_opt = kd_gkt_opt

    def get_smashed_data(self, net):
        smashed_data = []
        label = []
        net.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                fx_client = net(images)
                smashed_data.append(fx_client[-1].clone().detach().requires_grad_(True))
                label.append(labels)
        return smashed_data, label

    def train(self, net_client, net_ax, kd_logits = None):
        net_client.to(self.args.device)

        net_client.train()
        # Auxiliary net 분리 (클라이언트 쪽/서버 쪽)
        aux_client = copy.deepcopy(net_ax)

        params = list(net_client.parameters())
        
        for ax in aux_client:
            ax.to(self.args.device)
            ax.train()
            params += list(ax.parameters())

        optimizer_client = torch.optim.SGD(params, lr = self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        criterion = nn.CrossEntropyLoss() 
        criterion_KD =  SoftTarget(self.args.T)

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
                
                if self.kd_gkt_opt:
                    KD_loss_client = criterion_KD(client_logits, kd_logits[batch_idx].detach())
                    loss_client = loss_client + self.args.lambdaa * KD_loss_client

                batch_loss_c.append(loss_client.item())
                batch_acc_c.append(acc_client.item())    

                loss_client.backward()
                optimizer_client.step() 

            epoch_loss_c.append(sum(batch_loss_c)/len(batch_loss_c))
            epoch_acc_c.append(sum(batch_acc_c)/len(batch_acc_c))
        
        client_loss_acc = [sum(epoch_loss_c)/len(epoch_loss_c), epoch_acc_c[-1]]

        weight_a_c = [ax.state_dict() for ax in aux_client]

        # client-side logit 추출하기
        logits = []
        net_client.eval()
        aux_client[-1].eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args.device)
                fx_client = net_client(images)
                c_logit, _ = aux_client[-1](fx_client[-1])
                logits.append(c_logit.detach())
        # net_ax가 요소마다 데이트 되는지 확인해봐야 함
        return net_client.state_dict(), weight_a_c, logits, self.args, client_loss_acc[0], client_loss_acc[1] # sum(epoch_loss)/len(epoch_loss), epoch_acc[-1], sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]


class LocalUpdate_server(object):
    def __init__(self, args, smashed_data = None, label = None, wandb = None, model_idx = None, kd_gkt_opt = False):
        '''
        args : argument
        dataset : 학습에 사용될 데이터 셋
        idxs : 데이터 분할에 사용되는
        '''
        self.args = args 
        self.data = smashed_data
        self.label = label
        self.wandb = wandb
        self.model_idx = model_idx
        self.kd_gkt_opt = kd_gkt_opt
    def get_kd_logits(self, net):
        logits = []
        net.eval()
        with torch.no_grad():
            for i in range(len(self.data)):
                fx_server, probas = net(self.data[i])
                logits.append(fx_server[-1].detach())
        return logits


    def train(self, net, net_ax, kd_logits = None):
        net_server = copy.deepcopy(net)
        net_server.to(self.args.device)
        net_server.train()

        aux_server = copy.deepcopy(net_ax)

        params = list(net_server.parameters())
        
        for ax in aux_server:
            ax.to(self.args.device)
            ax.train()
            params += list(ax.parameters())

        optimizer_server = torch.optim.SGD(params, lr = self.args.lr_server, momentum = self.args.momentum, weight_decay = self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()  
        criterion_KD =  SoftTarget(self.args.T)

        
        net_server.train()
        net_server.zero_grad()
        optimizer_server.zero_grad()

        epoch_loss_s = []
        epoch_acc_s = []

        for iter in range(self.args.local_ep):
            batch_loss_s = []
            batch_acc_s = []

            for j in range(len(self.data)):
                fx = self.data[j]   # fx reshape 필요할듯?
                y = self.label[j]

                fx_server, probs = net_server(fx)
                logit, prob = net_ax[0](fx)
                loss = 0
                for i in range(1,len(fx_server)):
                    auxiliary_net = net_ax[i]# fx[i][1]-1]
                    auxiliary_net.zero_grad()
                    server_logits, server_probs = auxiliary_net(fx_server[i-1])
                    loss += criterion(server_logits, y)
                loss += criterion(fx_server[-1], y)
                loss += criterion(logit, y)
                acc = calculate_accuracy(fx_server[-1], y)  

                kd_loss = 0
                if self.kd_gkt_opt:
                    KD_loss_server = criterion_KD(fx_server[-1], kd_logits[j].detach())
                    loss = loss + KD_loss_server * self.args.lambdaa
                loss.backward()
                optimizer_server.step()

                batch_loss_s.append(loss.item())
                batch_acc_s.append(acc.item())
            epoch_loss_s.append(sum(batch_loss_s)/len(batch_loss_s))
            epoch_acc_s.append(sum(batch_acc_s)/len(batch_acc_s))
        server_loss = sum(epoch_loss_s)/len(epoch_loss_s)
        server_acc = epoch_acc_s[-1]
        
        weight_a_s = [ax.state_dict() for ax in aux_server]

        return net_server.state_dict(), weight_a_s,  self.args, server_loss, server_acc


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


class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss



    
