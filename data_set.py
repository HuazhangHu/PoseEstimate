import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import logging

logger = logging.getLogger(__name__)


class MyData(Dataset):

    def __init__(self, root_dir, child_dir):
        """
        :param root_dir: 数据集根目录
        :param child_dir: 数据集子目录
        """
        self.root_dir = root_dir
        self.child_dir = child_dir
        self.path = os.path.join(self.root_dir, self.child_dir)
        self.data_dir = os.listdir(self.path)

    def __getitem__(self, inx):
        """获取数据集中的item  """
        try:
            label_file_name = self.data_dir[inx]
            label_name = self.data_dir[inx].split('.')[0].split("_")[1]
            label = np.float32(int(label_name))
            item_path = os.path.join(self.root_dir, self.child_dir, label_file_name)
            datawash = DataWash(item_path)
            inputs = datawash.process()
            inputs_tensor = torch.Tensor(inputs)
            # sample = {'inputs': inputs, 'label': label}

            return inputs_tensor, label

        except Exception as e:
            logger.error("get item error : %s " % e)

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data_dir)

    def padding(self, sequences, max_len):
        max_size = max_len
        trailing_dims = max_size[1:]


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


def collate_fn(datas_label):
    datas_label.sort(key=lambda x: len(x[0]), reverse=True)
    datas_length = [len(sq[0]) for sq in datas_label]

    datas = [data[0] for data in datas_label]
    labels = [label[1] for label in datas_label]

    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, datas_length, torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # (-1,1) == (unknown,1)


class DataWash:
    def __init__(self, data_path):
        """
        :param data_path:json 输入数据json
        """
        self.data_path = data_path
        self.keyword = keyword

    def process(self):
        """
        :return: list[dist] 整个视频信息
        """
        try:
            with open(self.data_path, 'r') as load_f:
                data_dict = json.load(load_f)
                point_array = []
                flame_num = len(data_dict)
                for item in range(64):
                    # Nose = data_dict[item]['keypoints'][0:2]
                    # LHip = data_dict[item]['keypoints'][33:35]
                    # RHip = data_dict[item]['keypoints'][36:38]
                    # gravity = get_gravity(Nose, LHip, RHip)
                    # point_array.append(gravity)
                    key_points = np.array(data_dict[item]['keypoints'])
                    key_points = np.reshape(key_points, (17, 3))
                    l2_distance = self.pairwise_l2_distance(key_points)  # 每一帧的l2_distance (136,1)

                    # print(len(l2_distance))
                    point_array.append(l2_distance)
                point_array = np.array([point_array])  # (flame_num , 136)
                # point_array = torch.tensor(point_array)  # (flame_num , 136)
            return point_array

        except Exception as e:
            logger.error("data wash error : %s " % e)

    def get_l2_distance(self, a, b):
        """Computes pairwise distances between a point and b point."""
        dist = np.sqrt(np.sum((a - b) ** 2))
        return dist

    def pairwise_l2_distance(self, points):
        """遍历数组两两计算l2距离"""
        distance = []
        r = points.shape[0]  # 行数
        c = points.shape[1]  # 列数
        i = 0
        while i < r:
            j = i + 1
            while j < r:
                res = self.get_l2_distance(points[i], points[j])
                distance.append(res)
                j += 1
            i += 1

        return distance

# # data_set example
# root_dir = './data'
# label_dir = 'train'
# pose_dataset = MyData(root_dir, label_dir)
# trainloader = DataLoader(pose_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# print(len(trainloader))
# print(enumerate(trainloader))
# for batch_idx, batch_data in enumerate(trainloader):
#     print(type(batch_data))
#     print(len(batch_data['inputs']))
#     # print(batch_data['label'][0])
# #data_wash example
# datawash = DataWash('data/valid/1_10.json')
# data = datawash.process()
# print(data)
#
# # plot
# x = np.array(data)
# x_ = np.arange(0,len(data),1)
# y_ = x[:, 1]
# plt.plot(x_ , y_)
# plt.show()
