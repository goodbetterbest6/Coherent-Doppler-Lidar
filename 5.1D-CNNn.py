import torch
import torch.nn as nn
import numpy as np
# from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt
import time


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=20, stride=10, padding=4)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=20, stride=10, padding=4)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, stride=5, padding=5)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=21, stride=5, padding=5)
        self.pool5 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        # x = self.pool2(x)
        # x = torch.relu(self.conv3(x))
        # x = self.pool3(x)
        # x = torch.relu(self.conv4(x))
        # x = self.pool4(x)
        # x = torch.relu(self.conv5(x))
        x = self.pool5(x)
        print(x.shape)
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        # x = x.reshape(-1)
        print(x.shape)
        x = self.fc(x)
        return x


lossssssss = []
# lw = []
# lrl = [0.08, 0.05, 0.02, 0.01, 0.008, 0.005, 0.002, 0.001, 0.0008, 0.0005, 0.0002, 0.00001]


# 定义卷积神经网络模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建模型实例
model = MyCNN().to(device)

# 定义输入数据和目标数据
# batch_size = 10  # 设置batch_size
PATH = "data/"
train_set = np.load(PATH + "train_set.npy")
eval_set = np.load(PATH + "eval_set.npy")
# ti = train_input,tt=train_target,tei=test_input,tet=test_target
ti = train_set[:, 1:].astype(np.float32)
tt = train_set[:, 0].astype(np.float32)
tei = eval_set[:, 1:].astype(np.float32)
tet = eval_set[:, 0].astype(np.float32)
ti = np.transpose(ti.reshape(ti.shape[0], ti.shape[1], 1), (0, 2, 1))
test_input = np.transpose(tei.reshape(tei.shape[0], tei.shape[1], 1), (0, 2, 1))

input_data = torch.from_numpy(ti).to(device)
print(ti.shape)
target_data = torch.from_numpy(tt.reshape(tt.shape[0], 1)).to(device)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
min_loss = float('inf')
best_model_path = "1d-CNN_best_model.pth"
# 训练模型
num_epochs = 500

for epoch in range(num_epochs):
    optimizer.zero_grad()  # 梯度清零
    output = model(input_data)
    loss = loss_fn(output, target_data)
    loss.backward()
    optimizer.step()
    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), best_model_path)
    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# 使用训练好的模型进行预测
# 预测
model.load_state_dict(torch.load(best_model_path, map_location=device))
start_time = time.time()
test_input = torch.from_numpy(test_input).to(device)
prediction = model(test_input)
prediction = prediction.cpu().detach().numpy().reshape(-1)
end_time = time.time()
interval = end_time - start_time
print("1D-CNN time usage: {:.6f} seconds".format(interval))
print(type(prediction))
print(prediction.shape)
print(min_loss)
np.savetxt("1D_cnn_predict.txt", prediction)
plt.xlabel("real")
plt.ylabel("predict")
plt.scatter(prediction, tet)
plt.show()
