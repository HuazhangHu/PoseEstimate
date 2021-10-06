"""10.06 网络解构搭建"""

import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from data_set import MyData, logger, collate_fn
import torch.optim as optim

from transformer_model import MyModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import netron
import torch.onnx
from tensorboardX import SummaryWriter

root_dir = './data'
train_dir = 'train'
ckpt_dir = 'ckpt'
Batch_size = 16
NUM_FRAMES = 64
LR = 0.02

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

model = MyModel(num_frames=NUM_FRAMES)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
lossMAE = torch.nn.SmoothL1Loss()

print("model done !")

trainloader = DataLoader(MyData(root_dir, train_dir), batch_size=Batch_size, shuffle=True, collate_fn=collate_fn)

print("data done !")


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    print("Train: ")
    batch_idx = 0
    for inputs, datas_length, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        print(inputs.shape)
        period, periodicity = model(inputs)
        output = torch.sum(period, dim=1)
        train_loss = lossMAE(output, targets)
        train_loss.backward()
        optimizer.step()
        sample = inputs

        print(batch_idx, len(trainloader), 'Loss: %.3f ' % (train_loss / (batch_idx + 1)))
        batch_idx += 1

    # print('done')
    # with SummaryWriter(comment='MyNet', logdir='graph') as w:
    #     w.add_graph(model, (sample))

for i in range(1):
    train(i)
