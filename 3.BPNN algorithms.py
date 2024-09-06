#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np


# 设置路径,读取文件
PATH = 'data/'

##构建训练集和测试集
train_set = np.load(PATH + "train_set.npy")  # 训练集
eval_set = np.load(PATH + "eval_set.npy")  # 测试集

# 数据集进行最值归一化
train_min = np.min(train_set[:, 1:])
train_max = np.max(train_set[:, 1:])
train_set[:, 1:] = (train_set[:, 1:] - train_min) / (train_max - train_min)
eval_set[:, 1:] = (eval_set[:, 1:] - train_min) / (train_max - train_min)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("training process will operates on " + str(device))

# 转换数据
x_train, y_train, x_valid, y_valid = map(torch.tensor,
                                         (train_set[:, 1:], train_set[:, 0],
                                          eval_set[:, 1:], eval_set[:, 0]))

min_loss = 99999999999
input_size = x_train.shape[1]
hidden_size1 = x_train.shape[1]
hidden_size2 = x_train.shape[1]
output_size = 1
batch_size = x_train.shape[0]
lri = 0.002 # learning rate
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size1),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size1, hidden_size2),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size2, output_size),
).to(device)
cost = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(my_nn.parameters(), lr=lri)

losses = []
epoch = 3000
for i in np.arange(epoch):
    xx = x_train.float().clone().detach().requires_grad_(True).to(device)
    yy = y_train.float().clone().detach().requires_grad_(True).to(device)
    prediction = my_nn(xx).reshape(-1)
    loss = cost(prediction, yy)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    if min_loss > loss:
        torch.save(my_nn.state_dict(), "BPNN_best_model.pth")
        min_loss = loss
    optimizer.step()
    # 此处调整学习率
    # 打印损失
    losses.append(loss.cpu().data.numpy())
np.savetxt("losses_of_BPNN.txt", losses, delimiter=",")
# 计算预测结果
my_nn.load_state_dict(torch.load("BPNN_best_model.pth"))
xx_e = x_valid.float().clone().detach().requires_grad_(True).to(device)
predict = my_nn(xx_e).cpu().data.numpy()
# 保存结果
np.savetxt("BPNN_predict.txt", predict, delimiter=",")
np.savetxt("anemometer_values.txt", y_valid.data.numpy(), delimiter=",")
