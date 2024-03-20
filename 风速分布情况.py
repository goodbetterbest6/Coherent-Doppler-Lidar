import numpy as np
import matplotlib.pyplot as plt


def plot_and_save_distribution(data, resolution=0.1, file_name='wind_speed_distribution.txt'):
    # 绘制数值分布图
    plt.hist(data, bins=np.arange(0, np.max(data) + resolution, resolution), color='blue', alpha=0.7)
    plt.title('Number Distribution')
    plt.xlabel('Value Range')
    plt.ylabel('Count in Range')
    plt.grid(True)
    plt.show()

    # 计算每个区间内的数值量
    hist, bin_edges = np.histogram(data, bins=np.arange(0, np.max(data) + resolution, resolution))

    # 保存区间信息和数值量到文件
    with open(file_name, 'w') as file:
        for i in range(len(hist)):
            file.write(f"{bin_edges[i]:.1f}, {hist[i]}\n")


PATH = "data/"
data = np.load(PATH + "wind_radar.npy")
# print(data.shape)
data = data[:, 0]
# print(data)
plot_and_save_distribution(data)
