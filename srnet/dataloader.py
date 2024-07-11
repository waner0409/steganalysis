from torch.utils.data import Dataset, DataLoader
import numpy as np
from os import listdir
from os.path import join
import torch
from turbojpeg import TurboJPEG, TJCS_YCbCr
import random
# 创建TurboJPEG实例
jpeg = TurboJPEG()

class DataGenerator(Dataset):
    def __init__(self, cover_dir, stego_dir):
        self.cover_path = cover_dir
        self.stego_path = stego_dir

        cover_list = listdir(cover_dir)
        stego_list = listdir(stego_dir)
        self.filename_list = cover_list

        cover_len = len(cover_list)
        stego_len = len(stego_list)
        assert cover_len != 0, "the cover directory:{} is empty!".format(cover_dir)
        assert stego_len != 0, "the stego directory:{} is empty!".format(stego_dir)
        assert cover_len == stego_len, "the cover directory and stego directory don't have the same number files, " \
                                       "respectively： %d, %d" % (cover_len, stego_len)

        # 使用TurboJPEG读取第一张图片的Y通道确定图片尺寸
        with open(join(self.cover_path, self.filename_list[0]), 'rb') as file:
            data = file.read()
        height, width, _,_ = jpeg.decode_header(data)
        #print(height,width)#
        self.img_shape = (height, width)

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        batch = np.empty(shape=(2, self.img_shape[0], self.img_shape[1]), dtype='uint8')

        # 分别从封面和隐写图中读取Y通道
        for i, path in enumerate([self.cover_path, self.stego_path]):
            with open(join(path, self.filename_list[index]), 'rb') as file:
                data = file.read()
            planes = jpeg.decode_to_yuv_planes(data)
            y_plane = planes[0]  # 获取Y通道数据
            batch[i, :, :] = y_plane


        batch = np.expand_dims(batch, axis=3)  # 增加一个单通道维度
        label = torch.tensor([0, 1], dtype=torch.int64)

        rot = random.randint(0, 3)
        if random.random() < 0.5:
            #print([torch.from_numpy(np.rot90(batch, rot, axes=[1, 2]).copy()), label])
            return [torch.from_numpy(np.rot90(batch, rot, axes=[1, 2]).copy()), label]#[[[1024,1024,1],[1024,1024,1]],[0,1]]
        else:
            return [torch.from_numpy(np.flip(np.rot90(batch, rot, axes=[1, 2]).copy(), axis=2).copy()), label]

class DataGenerator1(Dataset):
    def __init__(self, data_dir):
        self.data_path = data_dir
        self.filename_list = listdir(data_dir)

        assert len(self.filename_list) != 0, "the directory: {} is empty!".format(data_dir)

        # 使用TurboJPEG读取第一张图片的Y通道确定图片尺寸
        with open(join(self.data_path, self.filename_list[0]), 'rb') as file:
            data = file.read()
        height, width, _, _ = jpeg.decode_header(data)
        self.img_shape = (height, width)

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        # 读取Y通道
        batch = np.empty(shape=(1, self.img_shape[0], self.img_shape[1]), dtype='uint8')
        with open(join(self.data_path, self.filename_list[index]), 'rb') as file:
            data = file.read()
        planes = jpeg.decode_to_yuv_planes(data)
        y_plane = planes[0]  # 获取Y通道数据
        batch[0,:, :] = y_plane

        batch = np.expand_dims(batch, axis=3)  # 增加一个批处理维度
        #print(batch.shape)
        return torch.from_numpy(batch)


def generate_data(data_path, batch_size):
    train_data = DataGenerator(data_path['train_cover'], data_path['train_stego'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size['train'], shuffle=True, num_workers=2, drop_last=True)

    valid_data = DataGenerator(data_path['valid_cover'], data_path['valid_stego'])
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size['valid'], drop_last=True)
    img, target = train_data[0]
    print(img.shape)
    print(target)
    return train_loader, valid_loader


def generate_test_data(data_path, batch_size):
    mytest_data = DataGenerator(data_path['test_cover'], data_path['test_stego'])
    valid_loader = DataLoader(dataset=mytest_data, batch_size=batch_size, drop_last=True)

    return valid_loader

def gt(data_path, batch_size):
    mytest_data = DataGenerator1(data_path)
    valid_loader = DataLoader(mytest_data, batch_size=batch_size, shuffle=False)

    return valid_loader
# data_path = {
#     'train_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/cover/',
#     'train_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/train/stego/',
#     'valid_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
#     'valid_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
# }
# batch_size = {'train': 8, 'valid': 8}
#
# train, valid = generate_data(data_path, batch_size)
#
# data_path = {
#     'test_cover': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/cover/',
#     'test_stego': '/home/dengruizhi/0.paper/3.datasets/1.dataset/WOW_BOSS_256_04/validation/stego/'
# }
# batch_size = 1
# test_data = generate_test_data(data_path, batch_size)
#
# for index, (images, labels) in enumerate(test_data):
#     print(images.shape)
#     print(labels.shape)
#     print(labels)
#     images = images.view(-1, 256, 256, 1)
#     labels = torch.squeeze(labels.view(-1, 1))  # 在网络中进行了NCHW转换
#     print(images.shape)
#     print(labels.shape)
#     print(labels)
#     break


