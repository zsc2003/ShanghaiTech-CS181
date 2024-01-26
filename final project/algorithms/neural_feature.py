import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import json

# 8*8 + 1
# board + turn

# output 1 or -1

# input data, board, turn, label: 1 or -1


# class feature_net(nn.Module):
#     def __init__(self):
#         super(feature_net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 3, 5)
#         self.resnet = models.resnet18(pretrained=True)
#         self.fc = nn.Linear(1000, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()

#     def forward(self, board, turn):
#         x = self.conv1(board)
#         x = self.relu(x)
#         x = self.resnet(x)
#         x = self.fc(x)
#         x = self.sigmoid(x)
#         return x

class feature_network(nn.Module):
    def __init__(self):
        super(feature_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=2)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.fc = nn.Linear(1000 + 1, 1)

    def forward(self, x, extra_input):
        # x: [batch_size, 1, 8, 8]
        # extra_input: [batch_size, 1]
        x = self.conv1(x)
        x = self.resnet(x)
        x = torch.cat((x.view(x.size(0), -1), extra_input), dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    

if __name__ == '__main__':
    data_path = r'C:\Users\wangquan\Desktop\workspace\cs181\ShanghaiTech-CS181-Final-Project\minimax_data\10000.json'
    # read
    # parse data
    json_file = open(data_path, 'r')
    data = json.load(json_file)
    data = data['data']
    net = feature_network()
    net.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 20
    for i in tqdm(range(num_epochs + 1)):
        for j in range(len(data)):
            # train
            
            # get board
            board = data[j]['board']
            # to tensor
            board = torch.tensor(board).float().cuda()
            # turn 8*8 to 8*8*1
            board = board.unsqueeze(0).unsqueeze(0)
            
            # get turn
            turn = data[j]['turn']
            # to tensor
            turn = torch.tensor(turn).float().cuda()
            turn = turn.view(1, 1)
            
            # get label
            label = data[j]['label']
            # to tensor
            label = torch.tensor(label).float().cuda()
            label = label.view(1, 1)
            
            # forward
            optimizer.zero_grad()
            output = net(board, turn)
            # loss
            # print("output", output)
            # print("label", label)
            loss = criterion(output, label)
            # backward
            loss.backward()
            optimizer.step()
        if i % 5 == 0:
            # save model every epoch record epoch number
            torch.save(net.state_dict(), os.path.join('ckpt', 'feature_net_epoch_{}.pth'.format(i)))