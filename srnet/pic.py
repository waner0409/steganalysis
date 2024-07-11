import os
import shutil
from sklearn.model_selection import train_test_split

def process_images(src_folder, dest_folders, split_ratios=(0.7, 0.15, 0.15)):
    images = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.jpg')]
    # 分割数据集
    train_files, test_files = train_test_split(images, test_size=sum(split_ratios[1:]), random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=split_ratios[2] / sum(split_ratios[1:]), random_state=42)

    # 分配文件到各个数据集文件夹
    folders = {
        'train': (train_files, dest_folders[0]),
        'val': (val_files, dest_folders[1]),
        'test': (test_files, dest_folders[2])
    }

    # 创建目标文件夹并复制文件
    for key, (file_list, folder) in folders.items():
        os.makedirs(folder, exist_ok=True)
        for file_path in file_list:
            # 直接复制图像文件到目标文件夹
            shutil.copy(file_path, os.path.join(folder, os.path.basename(file_path)))

# 定义源文件夹和目标文件夹路径
src_folders = ['./j_uniward/j_uniward_stego_5000', './j_uniward/j_uniward_cover_5000']
dest_folders = [
    ['./media/ste/tra', './media/ste/val', './media/ste/tst'],
    ['./media/cov/tra', './media/cov/val', './media/cov/tst']
]

# 处理stego和cover图像
for src, dest in zip(src_folders, dest_folders):
    process_images(src, dest)
