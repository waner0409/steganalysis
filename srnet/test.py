import torch
from dataloader import gt
#from train_valid_function import test
from SRNet import SRNet
import csv
import os

weight_path = './HUGO_01_Model/weight3/Model_27500.pth'#1050 3epoch 3500 10epoch

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True).to(device)
print('model down successfully')

data_path = './08'
files = os.listdir(data_path)

batch_size = 1
test_loader = gt(data_path, batch_size)
results = []
image_index = 0  # 初始化图像索引为0

model.load_state_dict(torch.load(weight_path))
model.eval()
with torch.no_grad():  # 不追踪梯度以加速计算并减少内存消耗
    for images in test_loader:
        images = images.view(-1, 1024, 1024, 1).to(device)
        #print(images.shape)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        results.append([files[image_index], predicted.item()])
        image_index += 1  # 更新图像索引

# 写入CSV文件
with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image_ID', 'Predicted_Label'])
    writer.writerows(results)

print('Model testing complete and results saved to predictions.csv')
