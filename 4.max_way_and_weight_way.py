import numpy as np
import matplotlib.pyplot as plt
import time

def weight_way(spec, bottom_rate):
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


PATH = 'data/'
eval_set = np.load(PATH + "eval_set.npy")[:, 1:1332]
speed = np.load(PATH + "eval_set.npy")[:, 0]
noise = np.load(PATH + "noise.npy")
noise = noise[:1331]
# 校正低频分量强度
start_time = time.time()
noise = noise - np.average(noise, axis=0) + 5
eval_set = eval_set - np.mean(eval_set, axis=1, keepdims=True) + 5
i_noise = np.tile(1 / noise, (eval_set.shape[0], 1))
eval_set = eval_set * i_noise
eval_set = eval_set - np.mean(eval_set, axis=1, keepdims=True)
w_speed = weight_way(eval_set, 0.96)
end_time = time.time()
interval = end_time - start_time
print("weight way time usage: {:.6f} seconds".format(interval))
eval_set = np.load(PATH + "eval_set.npy")[:, 1:1332]
noise = np.load(PATH + "noise.npy")
noise = noise[:1331]
start_time = time.time()
noise = noise - np.average(noise, axis=0) + 5
eval_set = eval_set - np.mean(eval_set, axis=1, keepdims=True) + 5
i_noise = np.tile(1 / noise, (eval_set.shape[0], 1))
eval_set = eval_set * i_noise
eval_set = eval_set - np.mean(eval_set, axis=1, keepdims=True)
max_speed = max_way(eval_set)
end_time = time.time()
interval = end_time - start_time
print("max way time usage: {:.6f} seconds".format(interval))
np.savetxt("max_way_predict.txt", max_speed)
np.savetxt("weight_way_predict.txt", w_speed)
plt.plot(w_speed)
plt.plot(max_speed)
plt.plot(speed)
plt.show()
