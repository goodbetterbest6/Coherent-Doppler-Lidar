import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time

# 定义LSTM回归模型
class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print(x.shape)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = "data/"
# 定义超参数
input_length = 1  # 输入数据长度
output_length = 1  # 输出数据长度
hidden_size = 10 # LSTM隐藏单元数量
num_layers = 10  # LSTM层数
epochs = 3000 # 迭代次数
learning_rate = 0.002  # 学习率
dropout_rate = 0  # 丢弃率
min_loss = float('inf')  # 设置为正无穷大

# 创建模型实例并移动到GPU（如果可用）
model = LSTMRegression(input_length, hidden_size, num_layers, output_length).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 准备训练集和测试集（示例数据）
ti = np.load(PATH + "train_set.npy")[:, 1:].astype(np.float32)
tt = np.load(PATH + "train_set.npy")[:, 0].astype(np.float32)
tei = np.load(PATH + "eval_set.npy")[:, 1:].astype(np.float32)
tet = np.load(PATH + "eval_set.npy")[:, 0].astype(np.float32)
train_input = torch.from_numpy(ti.reshape(ti.shape[0], ti.shape[1], 1)).to(device)
train_target = torch.from_numpy(tt.reshape(tt.shape[0], 1)).to(device)
test_input = torch.from_numpy(tei.reshape(tei.shape[0], tei.shape[1], 1)).to(device)
test_target = torch.from_numpy(tet.reshape(tet.shape[0], 1)).to(device)

# 进行训练
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()
    train_output = model(train_input)
    train_loss = criterion(train_output, train_target)
    train_loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, epochs, train_loss.item()))
    if min_loss > train_loss.item():
        min_loss = train_loss.item()
        # 保存模型
        torch.save(model.state_dict(), 'LSTM_best_model.pth')

# 使用训练好的模型进行预测（在测试集上）
model.load_state_dict(torch.load('LSTM_best_model.pth'))
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    test_output = model(test_input)
    test_loss = criterion(test_output, test_target)
    print('Test Loss: {:.4f}'.format(test_loss.item()))

# 保存预测结果并绘图
np.savetxt("LSTM_predict.txt", test_output.cpu().numpy())
plt.plot(test_output.cpu().numpy())
plt.plot(test_target.cpu().numpy())
plt.show()