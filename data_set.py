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

    # def padding(self, sequences, max_len):
    #     max_size = max_len
    #     trailing_dims = max_size[1:]


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


# def collate_fn(batch):
#     # 先排序 后padding
#     data = [item[0] for item in batch]
#     label = [item[1] for item in batch]
#     for index in range(len(data)):
#         data[index].sort(key=lambda x: len(x), reverse=True)
#         data[index] = pad_sequence(data[index], batch_first=True, padding_value=0)
#     data = np.array(data)
#     label = np.array(label)
#     # data.sort(key=lambda x: len(x), reverse=True)
#     # print(data.size())
#     # data = pad_sequence(data, batch_first=True, padding_value=0)
#
#     return [data,label]

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
        :return: list[dist] 包含每一帧图片信息的list
        """
        try:
            with open(self.data_path, 'r') as load_f:
                data_dict = json.load(load_f)
                point_array = []
                for item in range(len(data_dict)):
                    Nose = data_dict[item]['keypoints'][0:2]
                    LHip = data_dict[item]['keypoints'][33:35]
                    RHip = data_dict[item]['keypoints'][36:38]
                    gravity = get_gravity(Nose, LHip, RHip)
                    point_array.append(gravity)
                point_array = torch.tensor(point_array)
            return point_array

        except Exception as e:
            logger.error("data wash error : %s " % e)

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
# #
# # data_wash example
# datawash = DataWash('data/AlphaPose/alphapose-results2.json')
# data = datawash.process()
# print(data)
#
# # plot
# x = np.array(data)
# x_ = np.arange(0,len(data),1)
# y_ = x[:, 1]
# plt.plot(x_ , y_)
# plt.show()
