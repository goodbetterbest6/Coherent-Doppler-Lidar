import numpy as np
import matplotlib.pyplot as plt
import time

def centroid_way(spec, bottom_rate):
    spec = spec + 1
    speed = []
    # 重心法计算风速
    for i in np.arange(0, spec.shape[0], 1):  # 计算风速，重心法。
        # for i in np.arange(0,100,1):  #计算风速，重心法。
        Sumv = 0.0  # sum(x)
        Sumvx = 0.0  # sum(xf(x))
        Bottom = np.max(spec[i, :]) * bottom_rate  # 重心法的阈值，
        for j in np.arange(0, spec.shape[1], 1):
            if spec[i, j] > Bottom:
                Sumv = spec[i, j] + Sumv
                Sumvx = Sumvx + (spec[i, j] * (j))
        #             else:
        #                 pspec[i,j]=0.0#     print(Sumv,Sumvx)
        v = (Sumvx / Sumv) * 100 / 2048 * 1.55 / 2 if Sumv > 0 else 0  # 计算风速
        speed.append(v)
    #     print(spec_dataset[0,i])
    return speed


# 最大值法计算风速
def max_way(spec):
    speed = []
    for i in np.arange(0, spec.shape[0], 1):  # 遍历频谱数据
        v = (np.argmax(spec[i, :] + 1)) * 100 / 2048 * 1.55 / 2  # 将频谱数据的时间换成风速值
        speed.append(v)
    return speed

eval_set=np.load("./data/eval_set.npy")[:, 1:1332]
train_set = np.load("./data/train_set.npy")[:, 1:1332]
noise = np.load("./data/noise.npy")[:, 1:1332]
noise = np.mean(noise, axis = 0)
eval_set = eval_set - noise
train_set = train_set -noise
eval_set[:,0:32] = eval_set[:,0:32] - train_set[1150,0:32]
w_speed = centroid_way(eval_set, 0.9)
max_speed = max_way(eval_set)
np.savetxt("maximum_way_predict.txt", max_speed)
np.savetxt("centroid_way_predict.txt", w_speed)
plt.plot(w_speed)
plt.plot(max_speed)
plt.show()
