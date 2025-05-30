import hmm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from IOT.Optimizer import GA2

# 1.导入数据集
df_c1 = pd.read_csv('./data/data_c1.csv')
df_c4 = pd.read_csv('./data/data_c4.csv')
df_c6 = pd.read_csv('./data/data_c6.csv')

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

# 4.创建HMM模型
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
model.fit(data)

# 开始训练svr
module_pso.fit(X_train, Y_train)
module_ho.fit(X_train, Y_train)

# # 5.输出结果最佳超参数和评估指标
print("PSO的最优超参数：", best_c_pso, best_gamma_pso)
Y_pred_train = module_pso.predict(X_train)
Y_pred_test = module_pso.predict(X_test)
print("PSO训练集MSE:", mean_squared_error(Y_train, Y_pred_train))
# R2值表示模型解释的变异性的比例，范围从负无穷到1
# 接近1的R2值表示模型很好地解释了数据的变异性（即异常值）
print("PSO训练集R2:", r2_score(Y_train, Y_pred_train))
print("PSO测试集MSE:", mean_squared_error(Y_test, Y_pred_test))
print("PSO测试集R2:", r2_score(Y_test, Y_pred_test))

print('--------------------------------------------------------')

print("HO的最优超参数：", best_c_ho, best_gamma_ho)
Y_pred_train = module_ho.predict(X_train)
Y_pred_test = module_ho.predict(X_test)
print("HO训练集MSE:", mean_squared_error(Y_train, Y_pred_train))
# R2值表示模型解释的变异性的比例，范围从负无穷到1
# 接近1的R2值表示模型很好地解释了数据的变异性（即异常值）
print("HO训练集R2:", r2_score(Y_train, Y_pred_train))
print("HO测试集MSE:", mean_squared_error(Y_test, Y_pred_test))
print("HO测试集R2:", r2_score(Y_test, Y_pred_test))

