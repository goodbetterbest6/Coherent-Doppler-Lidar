import torch
import torch.nn as nn
import math
import numpy as np
from mealpy import FloatVar, SSA

class MyCNN(nn.Module):
    def __init__(self, out_channels1, kernel_size1, stride1, out_channels2, kernel_size2, stride2):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1,
                               padding=kernel_size1)
        out_lenth1 = math.floor((1331  +  kernel_size1) / stride1 + 1)
        out_lenth1 = out_lenth1 * out_channels1
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        out_lenth2 = math.floor(out_lenth1 / 2)
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2,
                               stride=stride2, padding=kernel_size2)
        out_lenth3 = math.floor( (out_lenth2 / out_channels1 + kernel_size2) / stride2 + 1)
        out_lenth3 = out_lenth3 * out_channels2
        self.fc = nn.Linear(int(out_lenth3), 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


def objective_function(solution):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solution = np.ceil(solution).astype(int)
    print(solution)
    out_channels1, kernel_size1, stride1, out_channels2, kernel_size2, stride2, learning_rate = solution
    learning_rate = learning_rate / 10000
    model = MyCNN(out_channels1, kernel_size1, stride1, out_channels2, kernel_size2, stride2).to(device)

    PATH = "data/"
    train_set = np.load(PATH + "train_set.npy")

    #归一化
    train_min = np.min(train_set[:, 1:])
    train_max = np.max(train_set[:, 1:])
    train_set[:, 1:] = (train_set[:, 1:] - train_min) / (train_max - train_min)

    train_input = train_set[:, 1:].astype(np.float32)
    train_target = train_set[:, 0].astype(np.float32)
    train_input = np.transpose(train_input.reshape(train_input.shape[0], train_input.shape[1], 1), (0, 2, 1))
    input_data = torch.from_numpy(train_input).to(device)
    target_data = torch.from_numpy(train_target.reshape(train_target.shape[0], 1)).to(device)
    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        # if epoch % 100 == 0:
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # 使用训练好的模型进行预测
    # 预测
    model.load_state_dict(torch.load(best_model_path))
    prediction = model(input_data)
    loss = loss_fn(prediction, target_data)
    return loss.cpu().detach().numpy()

problem_dict = {
    "bounds": FloatVar(lb=(1, 1, 1, 1, 1, 1, 1), ub=(32, 32, 32, 64, 32, 32, 10000)),
    "minmax": "min",
    "obj_func": objective_function,
    "log_to": "console",
}

model = SSA.DevSSA(epoch=50, pop_size=25, ST=0.8, PD=0.2, SD=0.1)
g_best = model.solve(problem_dict)
g_best.solution = np.ceil(g_best.solution).astype(int)
np.savetxt("SSA_result.txt", g_best.solution)