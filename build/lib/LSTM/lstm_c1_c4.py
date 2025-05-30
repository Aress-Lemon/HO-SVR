import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import Dataset, DataLoader

from IOT.LSTM.lstm import LSTM

# 设备 （cuda:0表示第一张显卡，如果有其他显卡，可以改变序号）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1.导入数据集
df_c1 = pd.read_csv('../data/data_c1.csv')
df_c4 = pd.read_csv('../data/data_c4.csv')
df_c6 = pd.read_csv('../data/data_c6.csv')

# 2.划分训练集(c1,c4)，测试集(c6)
# ignore_index=True: 忽略原来的索引，并重新生成一个新的连续的整数索引
df_c1_c4 = pd.concat([df_c1, df_c4], axis=0, ignore_index=True)
X_train = df_c1_c4.iloc[:, :-1]
Y_train = df_c1_c4.iloc[:, -1]
X_test = df_c6.iloc[:, :-1]
Y_test = df_c6.iloc[:, -1]

# 3.数据预处理
# ①使用MinMaxScaler对“时域提取的特征”进行归一化处理，缩放到[0,1]区间
sc = MinMaxScaler()
sc.fit(X_train)  # 计算均值，标准差
X_train = sc.transform(X_train) # 使用计算出的均值和标准差进行标准化
X_test = sc.transform(X_test) # 使用sc.fit(X_train)来标准化，而不使用sc.fit(X_test)，防止数据泄漏

# ②使用“滑移平均法”降低“时域提取的特征”的噪声
X_train = pd.DataFrame(X_train).rolling(window=11, min_periods=1).mean()
X_test = pd.DataFrame(X_test).rolling(window=11, min_periods=1).mean()

# ③使用PCA降维
pca = PCA(n_components=0.99)  # 保留99%的信息
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print(f'X_train形状：{X_train.shape}')
print(f'X_test形状：{X_test.shape}')
print(f'主成分个数：{pca.n_components_}')
print(f'各主成分的方差值占比：{pca.explained_variance_ratio_}')  # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
print(f'各主成分的方差值：{pca.explained_variance_}')  # 降维后的各主成分的方差值
print('--------------------------------------------------------')

# 4. 将数据集封装，方便使用DataLoader分批次处理
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

data_train = MyDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
data_test = MyDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))

trainLoader = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
testLoader = DataLoader(dataset=data_test, batch_size=64, shuffle=True)

# 5. 导入模型（import）、创建模型
module = LSTM(input_size=4, hidden_size=32, num_layers=2, output_size=1).to(device)

# 6. 定义损失函数、优化器、损失值记录数组
mse = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)

# 记录每一次批次(batch)训练的损失
train_loss = []
test_loss = []
# 记录每一次迭代的“平均损失”
train_epochs_loss = []
test_epochs_loss = []

# 7.开始训练
epoch_len = 100
for epoch in range(epoch_len):
    print('--------第 {} 轮训练开始-------'.format(epoch))

    # 5.1 训练模式
    module.train()
    train_epoch_loss = [] # 存放一个迭代(epoch)的所有损失==》用来计算一个迭代的“平均损失”
    for idx, (data_x, data_y) in enumerate(trainLoader):
        data_x = data_x.to(device)
        data_y = data_y.to(device)

        data_x = data_x.unsqueeze(1)  # 在中间扩充一个维度
        data_y = data_y.unsqueeze(-1).unsqueeze(-1)  # 连续扩充2个维度

        output = module(data_x)
        loss = mse(output, data_y)

        # 反向传播，更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        train_loss.append(loss.item())  # item: 转换为标量
        train_epoch_loss.append(loss.item())

        # 打印（只打印第一次，中间那一次“共两次”）
        if idx % (len(trainLoader)//2) == 0:
            print("epoch={}/{}, {}/{} of train, loss={}".format(epoch, epoch_len, idx, len(trainLoader), loss.item()))

    # 记录一次迭代的“平均损失”
    train_epochs_loss.append(np.average(train_epoch_loss))

    # 打印训练的“平均损失”
    print('平均训练损失：{}'.format(train_epochs_loss[-1]))

    # 5.2测试模式
    module.eval()
    test_epoch_loss = []  # 存放一个迭代的所有损失==》用来计算一个迭代的“平均损失”
    with torch.no_grad():   # 禁止计算梯度
        for idx, (data_x, data_y) in enumerate(testLoader):
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            data_x = data_x.unsqueeze(1)  # 在中间扩充一个维度（表示长度）
            data_y = data_y.unsqueeze(-1).unsqueeze(-1)  # 最后面扩充2个维度

            output = module(data_x)
            loss = mse(output, data_y)

            # 记录损失
            test_loss.append(loss.item())
            test_epoch_loss.append(loss.item())

        # 记录平均损失
        test_epochs_loss.append(np.average(test_epoch_loss))

        # 打印测试"平均损失"、整体正确率
        print('平均测试损失：{}'.format(test_epochs_loss[-1]))