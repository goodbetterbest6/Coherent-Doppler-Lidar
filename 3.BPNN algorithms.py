#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import math
import numpy as np
import time



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
lri = 0.005  # learning rate
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
    batch_loss = []
    xx = x_train.float().clone().detach().requires_grad_(True).to(device)
    yy = y_train.float().clone().detach().requires_grad_(True).to(device)
    prediction = my_nn(xx).reshape(-1)
    loss = cost(prediction, yy)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    if min_loss > loss:
        torch.save(my_nn.state_dict(), "BPNNbest.pth")
        min_loss = loss
        best_epoch = i
    optimizer.step()
    batch_loss.append(loss.cpu().data.numpy())
    # 此处调整学习率
    if (i + 1) % 25 == 0:
        optimizer.param_groups[0]['lr'] = lri * 0.95 ** (i / 200)

    # 打印损失
    if (i + 1) % 100 == 0:
        #         lri=lri*0.1
        #         optimizer = torch.optim.Adam(my_nn.parameters(), lr = lr)
        losses.append(np.mean(batch_loss))
        print(i + 1, np.mean(batch_loss), optimizer.param_groups[0]['lr'])
        print()
np.savetxt("losses.txt", losses, delimiter=",")
# 计算预测结果
my_nn.load_state_dict(torch.load("BPNNbest.pth"))
start_time = time.time()
xx_e = x_valid.float().clone().detach().requires_grad_(True).to(device)
predict = my_nn(xx_e).cpu().data.numpy()
end_time = time.time()
interval = end_time - start_time
print("BPNN time usage: {:.6f} seconds".format(interval))
# 画出图像
plt.xlabel("real")
plt.ylabel("predict")
plt.scatter(predict, y_valid.data.numpy())
plt.show()
n = 0
for i in np.arange(len(predict)):
    if abs(predict[i] - y_valid.data.numpy()[i]) > 1:
        n += 1
print("误差为1时能达到" + str(1 - n / len(predict)) + "准确率")
n = 0
for i in np.arange(len(predict)):
    if abs(predict[i] - y_valid.data.numpy()[i]) > 0.2:
        n += 1
print("误差为0.3时能达到" + str(1 - n / len(predict)) + "准确率")
print(n)
print("best_epoch",best_epoch)
# 保存结果
np.savetxt("BPNN_predict.txt", predict, delimiter=",")
np.savetxt("anemometer_value.txt", y_valid.data.numpy(), delimiter=",")
