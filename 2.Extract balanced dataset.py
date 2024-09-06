#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt


def build_set(wind_radar, vstart, vstep, split_num, split_data_num):
    speed_dataset = wind_radar[:, 0]
    spec_dataset = wind_radar[:, 1:]
    flag = 0
    k_max = spec_dataset.shape[0] - 1
    ready_set = np.zeros((split_num * split_data_num,
                          wind_radar.shape[1]),
                         dtype=float)  # 准备空数据集存放筛选出的数据
    for i in range(split_num):  # 划分区间
        n = i * split_data_num
        for k in range(len(speed_dataset)):  # 遍历数据集
            if vstart + vstep * (i + 1) >= speed_dataset[k] > vstart + vstep * i:  # 判断速度值是否在区间内
                ready_set[n, 1:] = spec_dataset[k, :]
                ready_set[n, 0] = speed_dataset[k]
                n += 1
            if n < (i + 1) * split_data_num and k == k_max:  # 判断当前区间数据量是否充足
                print("风速在" + str(vstart + vstep * (i)) + "到" + str(vstart + vstep * (i + 1)) + "区间上数据不足")
                print("还差" + str(split_data_num * (i + 1) - n) + "条数据")
            if n == (i + 1) * split_data_num:
                break
    print("数据集准备完毕")
    return ready_set

PATH= "data/"
wind_radar =np.load(PATH+"wind_radar.npy")

PATH = 'data/'
# 以速度为标准构建平衡数据集，vstart为速度起点，
# vstep为区间步长，split_num为区间个数，split_data_num为每个区间内数据。
v_start = 0.0
v_step = 0.1
split_num = 78
split_data_num = 30
ready_set = build_set(wind_radar, v_start, v_step, split_num, split_data_num)
train_set = ready_set[1::2, :]  # 训练集
eval_set = ready_set[0::2, :]
np.save(PATH + "train_set.npy", train_set)
np.save(PATH + "eval_set.npy", eval_set)
plt.plot(ready_set[:, 0])
print("程序执行完毕")
