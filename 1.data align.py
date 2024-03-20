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


def read_noise(dirpath, datalen):
    data = np.empty((0, datalen), dtype=float)
    for file in sorted(os.listdir(dirpath)):
        if os.path.isfile(dirpath + "/" + file):
            data = np.vstack((data, np.loadtxt(dirpath + "/" + file,
                                               dtype=float).reshape(-1, datalen)))
    spec_data = data[:, :-1]
    noise_a = np.average(spec_data, axis=0)
    return noise_a


# 时间内求平均
def average_data_by_time(time, data):
    data = data.reshape(data.shape[0], -1)
    unique_times = np.unique(time)  # 获取唯一的时间值
    averaged_data = np.zeros((unique_times.shape[0], data.shape[1]), dtype=float)  # 存储取平均数后的数据
    new_time = np.zeros_like(unique_times)  # 存储时间值
    j = 0
    for i, t in enumerate(unique_times):
        indices = np.where(time == t)  # 获取与当前时间对应的索引
        flag = 0
        for k in indices:
            if flag == 0:
                flag = 1
                temp_data = data[k, :]
            else:
                temp_data = np.vstack((temp_data, data[k, :]))
        averaged_data[j, :] = np.mean(temp_data, axis=0)  # 取平均数
        new_time[j] = t
        j += 1

    return new_time, averaged_data


# 将雷达时间转换为datetime64,精确度为s,向上取整.
def radar_time_to_real(float_value):
    flag = 0
    # 将float类型转换为字符串
    for i in float_value:
        float_str = str(i)
        # 解析日期部分（前八位）
        date_str = float_str[:8]
        # 构造日期部分字符串
        date_time_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        # 转换日期
        datetime_value = np.datetime64(date_time_str, "s")
        # 解析时间部分（小时、分钟、秒）并相加
        hour = int(float_str[8:10])
        #         datetime_value = datetime_value +np.timedelta64(int(hour), 'h')
        minute = int(float_str[10:12])
        #         datetime_value = datetime_value +np.timedelta64(int(minute), 'm')
        second = int(float_str[12:14]) + 60 * minute + 3600 * hour
        datetime_value = datetime_value + np.timedelta64(int(second) + 1, 's')
        if flag == 0:
            datetime_value_set = datetime_value
            flag = 1
        else:
            datetime_value_set = np.vstack((datetime_value_set, datetime_value))
    return datetime_value_set


def read_radar_dir(folder_path, data_len):
    flag = 0
    # 遍历文件夹内的所有文件
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.txt') and os.path.isfile(file_path):
            # 使用np.loadtxt()函数读取文件数据，并将数据存入数据列表
            file_data = np.loadtxt(file_path).reshape(-1, data_len)
            if flag == 0:
                data = file_data
                flag = 1
            else:
                data = np.vstack((data, file_data))
    print("雷达数据读取完毕")
    # 将数据列表转换为numpy数组
    data = np.array(data)
    radar_time = radar_time_to_real(data[:, -1])
    radar_spec = data[:, :-1]
    print("雷达数据转换完毕")
    return average_data_by_time(radar_time, radar_spec)


def read_wind_dir(folder_path):
    flag = 0
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.txt') and os.path.isfile(file_path):
            # 使用np.loadtxt()函数读取文件数据，并将数据存入数据列表
            file_data = np.loadtxt(file_path, delimiter=',',
                                   dtype="str", skiprows=1, )
            if flag == 0:
                data = file_data[:, 0:2]
                flag = 1
            else:
                data = np.vstack((data, file_data[:, 0:2]))
    print("风速计数据读取完毕")
    wind_time = data[:, 0].astype("datetime64[s]") + np.timedelta64(1, "s")
    wind_speed = data[:, 1].astype("float")
    return average_data_by_time(wind_time, wind_speed)


# 速度与频谱对应，通过时间
def compare_time(wind_time, wind_speed, radar_time, radar_spec):
    # 创建一个空数组用来存储匹配的值
    flag = 0
    # 遍历wind_time数组
    for i in range(len(wind_time)):
        # 在radar_time数组中查找与wind_time值相等的索引
        matching_indices = np.where(radar_time == wind_time[i])
        matching_indices = np.array(matching_indices)[0]
        # 检查是否找到匹配的索引
        if len(matching_indices) > 0:
            # 遍历所有匹配的索引，将wind_speed和radar_speed对应值存入matched_values数组
            for j in matching_indices:
                if flag == 0:
                    flag = 1
                    temp = np.hstack((wind_speed[i], radar_spec[j, :].reshape(-1))).reshape(-1)
                    matched_values = temp
                else:
                    temp = np.hstack((wind_speed[i], radar_spec[j, :].reshape(-1))).reshape(-1)
                    matched_values = np.vstack((matched_values, temp))
    # 将结果转换为NumPy数组并返回
    return matched_values


def read_noise(dirpath, datalen):
    data = np.empty((0, datalen), dtype=float)
    for file in sorted(os.listdir(dirpath)):
        if os.path.isfile(dirpath + "/" + file):
            data = np.vstack((data, np.loadtxt(dirpath + "/" + file,
                                               dtype=float).reshape(-1, datalen)))
    spec_data = data[:, :-1]
    noise_a = np.average(spec_data, axis=0)
    return noise_a


PATH = 'data/'
RADAR_PATH = PATH + "radar"  # 雷达数据路径
WIND_PATH = PATH + "wind"  # 风速计数据路径
NOISE_PATH = PATH + "noise"  # 噪声数据路径
DATA_LEN = 2049  # 每条雷达数据长度
print("读取目录完毕")
radar_time, radar_spec = read_radar_dir(RADAR_PATH, DATA_LEN)
print("time", radar_time.shape, "data", radar_spec.shape)
wind_time, wind_speed = read_wind_dir(WIND_PATH)
wind_radar = compare_time(wind_time, wind_speed, radar_time, radar_spec)[:, :1332]
np.save(PATH + "wind_radar", wind_radar)
noise = read_noise(NOISE_PATH,DATA_LEN)
np.save(PATH+"/noise.npy",noise)
print("程序执行完毕")
