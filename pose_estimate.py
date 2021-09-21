import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.utils.data import Dataset, DataLoader

from data_set import MyData, logger, collate_fn
import numpy as np


class Network(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_layer, n_classes, bias=True, batch_first=True, dropout=0.,
                 bidirectional=False):
        """
        :param in_dim: 输入单个数据维数
        :param hidden_dim: 隐藏层维数
        :param n_layer: LSTM叠加层数
        :param n_classes: 分类器
        """
        super(Network, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=batch_first, bias=bias,
                            dropout=dropout, bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, inputs):
        y_lstm, (h_n, c_n) = self.lstm(inputs)
        x = h_n[-1, :, :]
        x = self.classifier(x)
        return x


root_dir = './data'
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'
Batch_size = 16
INPUT_DIM = 2
HIDDEN_DIM = 32
LSTM_LAYERS = 2
OUT_DIM = 1
LR = 0.075

trainloader = DataLoader(MyData(root_dir, train_dir), batch_size=Batch_size, shuffle=True, collate_fn=collate_fn)
validloader = DataLoader(MyData(root_dir, valid_dir), batch_size=Batch_size, shuffle=True, collate_fn=collate_fn)
testloader = DataLoader(MyData(root_dir, test_dir), batch_size=1, shuffle=True, collate_fn=collate_fn)
# TODO :交叉验证
net = Network(in_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, n_layer=LSTM_LAYERS, n_classes=OUT_DIM)
net = net.to('cpu')
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print("Train: ")
    batch_idx = 0
    for inputs, datas_length, targets in trainloader:
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        optimizer.zero_grad()
        inputs_pack = pack_padded_sequence(inputs, datas_length, batch_first=True)
        outputs = net(inputs_pack)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_idx+=1
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def valid(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    print("Vaild: ")
    with torch.no_grad():
        for inputs, datas_length, targets in validloader:
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            inputs_pack = pack_padded_sequence(inputs, datas_length, batch_first=True)
            outputs = net(inputs_pack)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            predicted = outputs
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_idx+= 1
            print(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def Test():
    net.eval()
    print("Test: ")
    for inputs, datas_length, targets in validloader:
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        inputs_pack = pack_padded_sequence(inputs, datas_length, batch_first=True)
        outputs = net(inputs_pack)
        predicted =max(outputs.detach().numpy().reshape(1, -1)[0])
        print(predicted)
        # predicted = outputs.numpy()
        #
        # print(" Prediction : {0}".format(predicted))

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        valid(epoch)

        Test()
