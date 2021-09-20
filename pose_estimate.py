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

    def forward(self, input_tensor, seq_lens):
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        # TODO :padding 排序
        x_packed = pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        y_lstm, (h_n, c_n) = self.lstm(x_packed)
        y_padded = pad_packed_sequence(y_lstm, batch_first=self.batch_first, total_length=total_length)
        # 此时可以从out中获得最终输出的状态h
        # x = out[:, -1, :]
        x = h_n[-1, :, :]
        x = self.classifier(x)
        return x


root_dir = './data'
train_dir = 'data/train'
valid_dir = 'data/valid'
Batch_size = 1
trainloader = DataLoader(MyData(root_dir, train_dir), batch_size=Batch_size, shuffle=True, collate_fn=collate_fn)
validloader = DataLoader(MyData(root_dir, valid_dir), batch_size=Batch_size, shuffle=True, collate_fn=collate_fn)
# TODO :交叉验证
net = Network(2, 128, 1, 1)
net = net.to('cpu')
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print("Train: ")
    for batch_idx, batch_data in enumerate(trainloader):
        inputs = torch.Tensor([batch_data['inputs']])
        targets = torch.as_tensor(batch_data['label'], dtype=torch.float32)
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        optimizer.zero_grad()
        outputs = net(torch.squeeze(inputs, 1), (Batch_size,))
        outputs = torch.squeeze(outputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("Vaild: ")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(validloader):
            inputs = torch.Tensor([batch_data['inputs']])
            targets = torch.as_tensor(batch_data['label'], dtype=torch.float32)
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            outputs = net(torch.squeeze(inputs, 1), (Batch_size,))
            outputs = torch.squeeze(outputs, 1)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            predicted = outputs
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


for epoch in range(2):
    train(epoch)
    test(epoch)
