import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from data.dataset import DatasetSplit
from utils.utils import calculate_accuracy
import copy
import torch.nn.functional as F

class LocalUpdate(object):
    def __init__(self, args, p = 1.0, dataset = None, idxs = None):
        '''
        args : argument
        dataset : 학습에 사용될 데이터 셋
        idxs : 데이터 분할에 사용되는
        '''
        self.args = args
        self.p = p
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.bs, shuffle = True)
        self.selected_clients = []
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr = self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        #optimizer_client = torch.optim.Adam(net_client.parameters(), lr = self.args.lr, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss() 

        epoch_loss = []
        epoch_acc = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_acc = []
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                optimizer.zero_grad()

                # forward prop in client

                logits, probas = net(images) # torch.Size([32, 26, 8, 8])
                
                loss = criterion(logits, labels)
                acc = calculate_accuracy(logits, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]  # sum(epoch_loss)/len(epoch_loss), epoch_acc[-1], sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]

def test_img(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            logits, probas = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(probas, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = probas.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss


'''DepthFL'''
class LocalUpdate_d(object):
    def __init__(self, args, dataset = None, idxs = None, model_idx = None):
        '''
        args : argument
        dataset : 학습에 사용될 데이터 셋
        idxs : 데이터 분할에 사용되는
        '''
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.bs, shuffle = True)
        self.selected_clients = []
        self.model_idx = model_idx

    def train(self, net, net_ax):
        torch.autograd.set_detect_anomaly(True)
        net.to(self.args.device)
        net.train()

        params = list(net.parameters())

        for ax in net_ax:
            ax.to(self.args.device)
            ax.train()
            params += list(ax.parameters())

        optimizer = torch.optim.SGD(params, lr = self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss() 
        criterion_KD = SoftTarget(self.args.T)
        epoch_loss = []
        epoch_acc = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_acc = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                optimizer.zero_grad()

                # forward prop in client
                loss= 0
                KD_logits = []

                if self.model_idx < 3:        # full model은 포함되어 있지 않음
                    fx = net(images) # fx는 list형식
                    for i in range(len(fx)):
                        auxiliary_net = net_ax[i]
                        auxiliary_net.zero_grad()
                        client_logits, client_probs = auxiliary_net(fx[i])
                        KD_logits.append(client_logits)
                        loss += criterion(client_logits, labels)
                    acc = calculate_accuracy(client_logits, labels) 
                else:
                    fx, probs = net(images)
                    for i in range(len(fx)-1):
                        auxiliary_net = net_ax[i]
                        auxiliary_net.zero_grad()
                        client_logits, client_probs = auxiliary_net(fx[i])
                        KD_logits.append(client_logits)
                        loss += criterion(client_logits, labels)
                    loss += criterion(fx[-1], labels)
                    acc = calculate_accuracy(fx[-1], labels) 

                kd_loss = 0

                eps = 1e-8
                for i in range(len(KD_logits)):
                    for j in range(len(KD_logits)):
                        if  i == j:
                            continue
                        else:
                            kd_loss += criterion_KD(KD_logits[i], KD_logits[j])
                if len(KD_logits)!=0:
                    loss = loss + kd_loss/(len(KD_logits))
                else:
                    loss = loss + kd_loss
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        
        weight_a = [ax.state_dict() for ax in net_ax]
        return net.state_dict(), weight_a, sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]  # sum(epoch_loss)/len(epoch_loss), epoch_acc[-1], sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]

def test_img_d(net, datatest = None, args = None,  net_a = None):
    net.eval()
    net.to(args.device)
    
    if net_a:
        net_a.eval()
        net_a.to(args.device)

    # testing
    test_loss = 0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            if not net_a:
                logits, probas = net(data)
                test_loss += F.cross_entropy(probas, target, reduction='sum').item()
                # get the index of the max log-probability
                y_pred = probas.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            else:
                fx = net(data)
                # sum up batch loss
                logits, probas = net_a(fx[-1])
                test_loss += F.cross_entropy(probas, target, reduction='sum').item()
                # get the index of the max log-probability
                y_pred = probas.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

class SoftTarget(nn.Module):
    def __init__(self, T):  
        super(SoftTarget, self).__init__()
        self.T = T
        self.eps = 1e-9

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),F.softmax(out_t/self.T, dim=1)+ self.eps, reduction='batchmean') * self.T * self.T

        return loss
    

class LocalUpdate_sfl(object):
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


    def train(self, net_client, net_server):
        net_client.to(self.args.device)

        net_client.train()
        # Auxiliary net 분리 (클라이언트 쪽/서버 쪽)
        optimizer_client = torch.optim.SGD(net_client.parameters(), lr = self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        criterion = nn.CrossEntropyLoss() 

        epoch_loss_s = []
        epoch_acc_s = []

        for iter in range(self.args.local_ep):
            batch_loss_s = []
            batch_acc_s = []
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net_client.zero_grad()
                optimizer_client.zero_grad()

                # forward prop in client-side model      
                fx = net_client(images) # [[output1,depth1],[output2,depth2],...]
                fx_client = fx.clone().detach().requires_grad_(True)

                dfx, net_server, loss_s, acc_s = train_server(fx_client, net_server, labels, self.args, self.args.device)

                fx.backward(dfx)
                optimizer_client.step() 

                batch_loss_s.append(loss_s.item())
                batch_acc_s.append(acc_s.item())  

            epoch_loss_s.append(sum(batch_loss_s)/len(batch_loss_s))
            epoch_acc_s.append(sum(batch_acc_s)/len(batch_acc_s))
        
        loss_acc = [sum(epoch_loss_s)/len(epoch_loss_s), epoch_acc_s[-1]]

        # net_ax가 요소마다 데이트 되는지 확인해봐야 함
        return net_client.state_dict(), net_server.state_dict(), self.args, loss_acc[0], loss_acc[1] # sum(epoch_loss)/len(epoch_loss), epoch_acc[-1], sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]


def train_server(fx_client, net_s, y, args, device):
    net_server = copy.deepcopy(net_s).to(device)
    net_server.train()
    criterion = nn.CrossEntropyLoss() 

    optimizer_server = torch.optim.SGD(net_server.parameters(), lr = args.lr, momentum= args.momentum, weight_decay=args.weight_decay)

    fx = fx_client.to(device)
    y = y.to(device)

    fx_server,_ = net_server(fx)

    loss = criterion(fx_server,y)
    acc = calculate_accuracy(fx_server, y)

    loss.backward()
    dfx_client = fx.grad.clone().detach()
    optimizer_server.step()

    return dfx_client, copy.deepcopy(net_server), loss, acc



def test_img_sfl(net_c, net_s, datatest, args):
    net_c.eval()
    net_c.to(args.device)

    net_s.eval()
    net_s.to(args.device)


    # testing
    test_loss_s = 0 
    correct_s = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader) # len(data_loader)= 469개, 128*468개 세트, 1개는 96개 들어있음

    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):
            if 'cuda' in args.device:
                data, target = data.to(args.device), target.to(args.device)
            fx_client = net_c(data)

            # server-side loss
            logits_s, probas_s = net_s(fx_client)

            test_loss_s += F.cross_entropy(probas_s, target, reduction='sum').item()
            y_pred_s = probas_s.data.max(1, keepdim=True)[1]     # label return 
            correct_s += y_pred_s.eq(target.data.view_as(y_pred_s)).long().cpu().sum()  # ([32]) -> ([32, 1])

    test_loss_s /= len(data_loader.dataset)
    accuracy_s = 100.00 * correct_s / len(data_loader.dataset)

    return  accuracy_s, test_loss_s