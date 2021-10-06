import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
logger = logging.getLogger(__name__)
keyword = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
           "LHip", "RHip", "LKnee", "Rknee", "LAnkle", "RAnkle"]


def get_gravity(point_a, point_b, point_c):
    """
    直角坐标系计算三角形重心
    :param 三个点的坐标
    :return: [x,y]
    """
    x_ = (point_a[0] + point_b[0] + point_c[0]) / 3
    y_ = (point_a[1] + point_b[1] + point_c[1]) / 3
    gravity = [x_, y_]
    return gravity


def find_apex(x_, y_, flag=0):
    """
    :param flag: 单双脚
    :param x_:
    :param y_:list[y]
    :return:
    """
    y_min = min(y_)
    y_max = max(y_)
    distance = None  # 相邻峰值之间的最小水平距离，越小去噪越严格
    if flag == 1:  # 单脚
        height_base = y_min + (y_max - y_min) * 0.475  # 峰值的最小高度
    else:  # 双脚
        height_base = y_min + (y_max - y_min) * 0.65  # 峰值的最小高度
    prominence = (y_max - height_base) * 0.5  # 相邻两个波峰的最小突起程度，越小去噪越严格
    peaks, properties = find_peaks(y_, height=height_base, distance=distance, prominence=prominence)

    # print(" Apex : ", len(peaks))
    # print(peaks)
    # print(properties)
    # plt.plot(x_, y_)
    # plt.plot(peaks, y_[peaks], color='red')
    # plt.show()
    # num_peaks = len(peaks)

    return peaks


class RepCount:
    def __init__(self, data_path):
        """
        :param data_path:json 输入数据json
        """
        self.data_path = data_path
        self.keyword = keyword
        self.posture = 0  # 默认设置双脚跳

    def run(self):
        """
        :return: list[dist] 包含每一帧图片信息的list
        """
        try:
            with open(self.data_path, 'r') as load_f:
                data_dict = json.load(load_f)
                Ankles = []  # 脚踝
                Knees = []  # 膝盖
                Gravity = []
                for item in range(len(data_dict)):
                    Nose = data_dict[item]['keypoints'][0:2]
                    LHip = data_dict[item]['keypoints'][33:35]
                    RHip = data_dict[item]['keypoints'][36:38]
                    LAnkle = data_dict[item]['keypoints'][45:47]
                    RAnkle = data_dict[item]['keypoints'][48:50]
                    LKnee = data_dict[item]['keypoints'][39:41]
                    RKnee = data_dict[item]['keypoints'][42:44]
                    gravity = get_gravity(Nose, LHip, RHip)

                    Ankles.append([LAnkle[1], RAnkle[1]])
                    Knees.append([LKnee[1], RKnee[1]])
                    Gravity.append(gravity[1])

                if self.judge_posture(Ankles) and self.judge_posture(Knees):
                    # 判断为单脚跳
                    self.posture = 1
                    points = np.array(Ankles)
                    x_ = np.arange(0, len(points), 1)
                    y_left = points[:, 0]
                    y_right = points[:, 1]
                    y_abs = abs(np.array(y_left) - np.array(y_right))
                    peaks = find_apex(x_, y_abs, self.posture)
                    print("single foot num: ", len(peaks))
                    return peaks
                else:
                    self.posture = 0
                    points = np.array(Ankles)
                    x_ = np.arange(0, len(points), 1)
                    y_left = points[:, 0]
                    y_right = points[:, 1]
                    l_peaks = find_apex(x_, y_left, self.posture)
                    r_peaks = find_apex(x_, y_right, self.posture)
                    total = find_apex(x_, Gravity, self.posture)
                    # print('left feet: ', len(l_peaks))
                    # print('right feet: ', len(r_peaks))
                    print('double foot jump total : ', len(total))
                    return total

        except Exception as e:
            logger.error("data wash error : %s " % e)

    def judge_posture(self, pairs):
        """判断是双脚跳还是单脚跳
        :pairs 双膝或双脚踝
        :return 1单脚跳, 0 双脚跳
        """
        l_pair = np.array(pairs)[:, 0]
        r_pair = np.array(pairs)[:, -1]
        x_ = np.arange(0, len(pairs), 1)
        l_peaks = find_apex(x_, l_pair, flag=0).tolist()
        r_peaks = find_apex(x_, r_pair, flag=0).tolist()
        peroid_index = []
        temp = l_peaks+r_peaks
        temp.sort()
        i = 0
        j = 1

        while j < len(temp):
            if abs(temp[i]-temp[j]) <= 1:
                peroid_index.append(temp[i])
            i += 1
            j += 1
        if len(peroid_index) > 3:
            # print("从 {0} 帧到 {1} 帧都是在双脚跳".format(peroid_index[0], peroid_index[-1]))
            return 0
        else:
            return 1



def expriment(path):
    file_list = os.listdir(path)
    acc1, acc2, acc3 = 0, 0, 0
    for file in file_list:
        if file.split('.')[0].split('_')[1]:
            target = int(file.split('.')[0].split('_')[1])
            Counter = RepCount(os.path.join(path, file))
            num_peaks = len(Counter.run())

            if abs(num_peaks - target) <= 1:
                acc1 += 1
            if abs(num_peaks - target) <= 2:
                acc2 += 1
            if abs(num_peaks - target) <= 3:
                acc3 += 1
    print('Accuracy in error 1 : {0}%  ({1}/{2})'.format(100. * acc1 / len(file_list), acc1, len(file_list)))
    print('Accuracy in error 2 : {0}%  ({1}/{2})'.format(100. * acc2 / len(file_list), acc2, len(file_list)))
    print('Accuracy in error 3 : {0}%  ({1}/{2})'.format(100. * acc3 / len(file_list), acc3, len(file_list)))

# # example for single json
file_test = "data/temp/29_129.json"
# Counter = RepCount(os.path.join(file_test))
# peaks = Counter.run()
# print(peaks)


# #example for batch file
path = r'data\train'
expriment(path)

