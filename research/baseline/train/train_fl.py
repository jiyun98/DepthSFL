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
                if self.args.kd_opt:
                    for i in range(len(KD_logits)):
                        for j in range(len(KD_logits)):
                            if  i == j:
                                continue
                            else:
                                kd_loss += criterion_KD(KD_logits[i], KD_logits[j])
                    loss = loss + kd_loss/(len(KD_logits))
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