import numpy as np
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt

def calculate_correlation(true_values, predicted_values):
    n = len(true_values)
    mean_true = sum(true_values) / n
    mean_predicted = sum(predicted_values) / n

    covariance = sum((true_values[i] - mean_true) * (predicted_values[i] - mean_predicted) for i in range(n))
    variance_true = sum((x - mean_true) ** 2 for x in true_values)
    variance_predicted = sum((x - mean_predicted) ** 2 for x in predicted_values)

    correlation = covariance / math.sqrt(variance_true * variance_predicted)
    return correlation


def calculate_std_i(true, predict):
    return math.sqrt(sum((true - predict) * (true - predict)) / len(true))


PATH = "result/"
real_PATH = PATH + "anemometer_value.txt"
predict_PATH =  "1D_cnn_predict.txt"
# 真实值和预测值
true_values = np.loadtxt(real_PATH)
predicted_values = np.loadtxt(predict_PATH)

# 计算相关系数和p-value
correlation, p_value = pearsonr(true_values, predicted_values)
print(predict_PATH)
print("相关系数", correlation)
# print(calculate_correlation(true_values,predicted_values))
print("标准差", calculate_std_i(true_values, predicted_values))
print("最大绝对误差", max(abs(true_values - predicted_values)))
print("平均绝对误差", sum(abs(predicted_values - true_values)) / len(predicted_values))
print("均方误差", sum((predicted_values - true_values) * (predicted_values - true_values)) / len(predicted_values))
n = 0
for i in np.arange(len(predicted_values)):
    if abs(predicted_values[i] - true_values[i]) > 0.3:
        n += 1
print("误差为0.3时能达到" + str(1 - n / len(predicted_values)) + "准确率")
print("n", n)
plt.plot(predicted_values)
plt.show()