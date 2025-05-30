import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

if __name__ == '__main__':
    """1.测试是否能取到某一列>165的行号"""
    # data_rawy = pd.read_csv('./raw/c1_wear.csv')
    #
    # # any(axis=1):  检查这一行是否有一个元素为True
    # # data_rawy.index： 取下标
    # row_wear = data_rawy.index[(data_rawy.iloc[:, 1:] > 165).any(axis=1)][0]
    #
    # print(row_wear)

    """2.测试一下是否能导入数据data_c1"""
    # df_c1 = pd.read_csv('./data/data_c1.csv')
    # print(df_c1.head())
    # print(f"列数: {len(df_c1.columns)}")
    # # 移除 'Unnamed: 0' 列
    # df = df_c1.drop(columns=['Unnamed: 0'])
    # print(df.head())
    # df.to_csv('./data/data_c1.csv', index=False)

    # df_c4 = pd.read_csv('./data/data_c4.csv')
    # print(df_c4.head())
    # print(f"列数: {len(df_c4.columns)}")
    # # 移除 'Unnamed: 0' 列
    # df = df_c4.drop(columns=['Unnamed: 0'])
    # print(df.head())
    # df.to_csv('./data/data_c4.csv', index=False)

    # df_c6 = pd.read_csv('./data/data_c6.csv')
    # print(df_c6.head())
    # print(f"列数: {len(df_c6.columns)}")
    # # 移除 'Unnamed: 0' 列
    # df = df_c6.drop(columns=['Unnamed: 0'])
    # print(df.head())
    # df.to_csv('./data/data_c6.csv', index=False)

    # 3.测试将两个dataframe行拼接
    # df_c1 = pd.read_csv('./data/data_c1.csv')
    # df_c4 = pd.read_csv('./data/data_c4.csv')
    # df_c6 = pd.read_csv('./data/data_c6.csv')
    #
    # df_c1_c4 = pd.concat([df_c1, df_c4], axis=0, ignore_index=True)
    # X_train = df_c1_c4.iloc[:, :-1]
    # Y_train = df_c1_c4.iloc[:, -1]
    # print(X_train.head())
    # print(Y_train.head())

    # 4. 主层次分析法PCA
    A = np.array(
        [[84, 65, 61, 72, 79, 81],
         [64, 77, 77, 76, 55, 70],
         [65, 67, 63, 49, 57, 67],
         [74, 80, 69, 75, 63, 74],
         [84, 74, 70, 80, 74, 82]])

    # pca = PCA(n_components=2)  # 降到 2 维
    pca = PCA(n_components=0.9)  # 保留90%的成分

    pca.fit(A)  # 用A拟合模型
    new_A = pca.transform(A) # 对A降维

    print(new_A)
    print(pca.explained_variance_ratio_) # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
    print(pca.explained_variance_) # 降维后的各主成分的方差值
