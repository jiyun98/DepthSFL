import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from data.dataset import DatasetSplit
from utils.utils import calculate_accuracy
import copy
import torch.nn.functional as F

class LocalUpdate(object):
    def __init__(self, args, dataset = None, idxs = None):
        '''
        args : argument
        dataset : 학습에 사용될 데이터 셋
        idxs : 데이터 분할에 사용되는
        '''
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.local_bs, shuffle = True)
        self.selected_clients = []
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr = self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        #optimizer_client = torch.optim.Adam(net_client.parameters(), lr = self.args.lr, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss() 

        epoch_loss_s = []
        epoch_acc_s = []

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

            epoch_loss_s.append(sum(batch_loss)/len(batch_loss))
            epoch_acc_s.append(sum(batch_acc)/len(batch_acc))
        
        return net.state_dict(),  sum(epoch_loss_s)/len(epoch_loss_s),epoch_acc_s[-1]  # sum(epoch_loss)/len(epoch_loss), epoch_acc[-1], sum(epoch_loss)/len(epoch_loss), epoch_acc[-1]

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
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
