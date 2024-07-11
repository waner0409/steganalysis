import torch
from dataloader import generate_test_data
from train_valid_function import test
from SRNet import SRNet


weight_path = './HUGO_01_Model/weight/Model_600.pth'

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

# 数据预处理
data_path = {
    'test_cover': './media/cov/tst',
    'test_stego': './media/ste/tst'
}
batch_size = 1
test_loader = generate_test_data(data_path, batch_size)
print('data_loader down successfully')

test(model=model,
     test_loader=test_loader,
     device=device,
     weight_path=weight_path)


