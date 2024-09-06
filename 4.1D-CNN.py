import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt


class MyCNN(nn.Module):
    def __init__(self, out_channels1, kernel_size1, stride1, out_channels2, kernel_size2, stride2):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1,
                               padding=kernel_size1)
        out_lenth1 = math.floor((1331 + kernel_size1) / stride1 + 1)
        out_lenth1 = out_lenth1 * out_channels1
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        out_lenth2 = math.floor(out_lenth1 / 2)
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2,
                               stride=stride2, padding=kernel_size2)
        out_lenth3 = math.floor((out_lenth2 / out_channels1 + kernel_size2) / stride2 + 1)
        out_lenth3 = out_lenth3 * out_channels2
        self.fc = nn.Linear(int(out_lenth3), 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


# 设置GPU运算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 导入SSA参数并构建网络
solution = np.loadtxt("./result/1.SSA_result.txt")
solution = solution.astype(int)
print(solution)
out_channels1, kernel_size1, stride1, out_channels2, kernel_size2, stride2, learning_rate = solution
learning_rate = learning_rate / 10000

model = MyCNN(out_channels1, kernel_size1, stride1, out_channels2, kernel_size2, stride2).to(device)

PATH = "data/"
train_set = np.load(PATH + "train_set.npy")
eval_set = np.load(PATH + "eval_set.npy")

# 归一化
train_min = np.min(train_set[:, 1:])
train_max = np.max(train_set[:, 1:])
train_set[:, 1:] = (train_set[:, 1:] - train_min) / (train_max - train_min)
eval_set[:, 1:] = (eval_set[:, 1:] - train_min) / (train_max - train_min)
# 分离风速和雷达采样值
train_input = train_set[:, 1:].astype(np.float32)
train_target = train_set[:, 0].astype(np.float32)
test_input = eval_set[:, 1:].astype(np.float32)
test_target = eval_set[:, 0].astype(np.float32)

train_input = np.transpose(train_input.reshape(train_input.shape[0], train_input.shape[1], 1), (0, 2, 1))
test_input = np.transpose(test_input.reshape(test_input.shape[0], test_input.shape[1], 1), (0, 2, 1))
test_input = torch.from_numpy(test_input).to(device)

input_data = torch.from_numpy(train_input).to(device)
target_data = torch.from_numpy(train_target.reshape(train_target.shape[0], 1)).to(device)
# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
min_loss = float('inf')
best_model_path = "1d-CNN_best_model.pth"
# 训练模型
num_epochs = 3000
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()  # 梯度清零
    output = model(input_data)
    loss = loss_fn(output, target_data)
    loss.backward()
    losses.append(loss.cpu().detach().numpy().reshape(-1))
    optimizer.step()
    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), best_model_path)
    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# 使用训练好的模型进行预测
# 预测
model.load_state_dict(torch.load(best_model_path, map_location=device))
prediction = model(test_input)
prediction = prediction.cpu().detach().numpy().reshape(-1)
np.savetxt("1D_cnn_predict.txt", prediction)
np.savetxt("Loss_of_1DCNN.txt", losses)
np.savetxt("anemometer_values.txt", test_target)