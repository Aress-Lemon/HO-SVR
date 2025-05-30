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

# 4.创建4个模型进行对照实验
bounds = np.array([[0.01, 0.01],[100.0, 100.0]])
dimension = 2 # 要优化的参数个数

def fitness(params):
    C, gamma = params
    svm_model = SVR(kernel='rbf', gamma=gamma, C=C)

    # 负均方误差
    scores = cross_val_score(
        svm_model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1  # 并行加速
    )
    mse = -np.mean(scores) # 转换为mse

    return mse


# ③PSO优化参数的SVR模型
from Optimizer import PSO_SVR

n_particles = 100 # 粒子个数
n_iteration_pso = 70 # 轮次
options_pso = {'c1': 1.6, 'c2': 1.8, 'w': 0.9} # 如惯性权重w、个体学习因子c1、群体学习因子c2等参数

best_c_pso, best_gamma_pso = PSO_SVR.optimize_svm(n_particles, n_iteration_pso, bounds, dimension, fitness, options_pso)

module_pso = SVR(kernel='rbf', gamma=best_gamma_pso, C=best_c_pso)

# ④HO优化参数的SVR模型
from Optimizer import HO_SVR

n_hippo = 16 # 河马个数
n_iteration_ho = 50 # 轮次
# 感觉参数边界最难
lower_bound = np.array([0.01, 0.01])
upper_bound = np.array([100.0, 100.0])

best_c_ho, best_gamma_ho = HO_SVR.HO(n_hippo, n_iteration_ho, lower_bound, upper_bound, dimension, fitness)

module_ho = SVR(kernel='rbf', gamma=best_gamma_ho, C=best_c_ho)

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

