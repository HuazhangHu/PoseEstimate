"""找峰值"""
import os
from data_set import DataWash
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def find_apex(data):
    points = np.array(data)
    x_ = np.arange(0, len(points), 1)
    y_ = points[:, 1]
    y_min = min(y_)
    y_max = max(y_)
    distance = None  # 相邻峰值之间的最小水平距离，越小去噪越严格
    height_base = y_min + (y_max - y_min) * 0.66  # 峰值的最小高度
    prominence = (y_max - height_base) * 0.5  # 相邻两个波峰的最小突起程度，越小去噪越严格
    peaks, properties = find_peaks(y_, height=height_base, distance=distance, prominence=prominence)
    # print(" Apex : ", len(peaks))
    # print(peaks)
    # print(properties)
    # plt.plot(x_, y_)
    # plt.plot(peaks, y_[peaks], color='red')
    # plt.show()
    num_peaks = len(peaks)

    return num_peaks


path = r'D:\PoseEstimate\data\train'

file_list = os.listdir(path)
acc1, acc2, acc3 = 0, 0, 0
for file in file_list:
    if file.split('.')[0].split('_')[1]:
        target = int(file.split('.')[0].split('_')[1])
        datawash = DataWash(os.path.join(path, file))
        data = datawash.process()
        num_peaks = find_apex(data)
        if abs(num_peaks-target) <= 1:
            acc1 += 1
        if abs(num_peaks - target) <= 2:
            acc2 += 1
        if abs(num_peaks - target) <= 3:
            acc3 += 1
print('Accuracy in error 1 : {0}%  ({1}/{2})'.format(100.*acc1/len(file_list), acc1, len(file_list)))
print('Accuracy in error 2 : {0}%  ({1}/{2})'.format(100.*acc2/len(file_list), acc2, len(file_list)))
print('Accuracy in error 3 : {0}%  ({1}/{2})'.format(100.*acc3/len(file_list), acc3, len(file_list)))