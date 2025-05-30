#coding=gbk

import pandas as pd
import numpy as np
import os

class DataProcess:
    def __init__(self, path):
        """
        初始化函数，设置数据文件路径，并记录开始时间。
        :param path: 数据文件夹路径
        """
        self.path = path  # 存储数据文件路径

    def get_all_file(self):
        """
        获取文件夹中的所有文件名称
        :return: 文件列表
        """
        files = os.listdir(self.path)
        # files.sort(key=lambda x: int(x[:-4]))  # 按文件名排序（去掉后四位）
        s = []
        for file in files:
            if not os.path.isdir(os.path.join(self.path, file)):  # 判断是否为文件
                filename = os.path.join(self.path, file)  # 拼接上路径
                s.append(filename)
        return s

    def get_data(self, i):
        """
        根据"索引"读取对应的CSV文件内容。
        :param i: 文件索引
        :return: DataFrame格式的数据
        """
        list_files = self.get_all_file()
        data = pd.read_csv(list_files[i-1])
        return data

    def get_shape(self, n=2):
        """
        获取前n个csv文件的形状（行数和列数）并打印。
        :param n: 文件数量
        """
        print('-------start output shape------')
        for i in range(1, n+1):
            list_files = self.get_all_file()
            data = pd.read_csv(list_files[i-1])
            print(f'{i} shape is: {data.shape}', end=' ; ')
            if i % 3 == 0:
                print()
        print('-------output done------')

    def get_max_min_shape(self, n=2):
        """
        输出前n个csv文件的最大、最小和平均行数。
        :param n: 文件数量
        """
        print('-------start output shape------')
        list_files = self.get_all_file()
        k = [pd.read_csv(list_files[i]).shape[0] for i in range(n)]
        k = np.array(k)
        max_val = np.max(k)
        min_val = np.min(k)
        mean_val = np.mean(k)
        print(f'max : {max_val}, min: {min_val}, mean: {mean_val}')
        print('-------output done------')

    def get_shape_num(self, n=2):
        """
        计算前n个csv文件的所有行数总和并打印。
        :param n: 文件数量
        """
        print('-------start output shape number------')
        total_rows = 0
        for i in range(1, n+1):
            list_files = self.get_all_file()
            data = pd.read_csv(list_files[i-1])
            total_rows += data.shape[0]
        print(f'all shape rows sum is : {total_rows}')
        print('-------output shape number done------')

    def get_file_feature(self, i=1):
        """
        提取单个CSV文件的10个统计特征。
        其中csv文件中每一列分别对应x、y、z方向切削力； x、y、z方向的震动； AE-RMS (V)
        :param i: 文件索引
        :return: 特征列表
        """

        data = self.get_data(i)  # 取出第i个csv文件

        # 存放特征
        li = []

        # 1.x方向上切削力的“标准差”
        data_1 = data.iloc[:, 0]
        std_fx = data_1.std()  # 标准差
        li.append(std_fx)

        # 2.y方向切削力的“峰峰值”
        data_2 = data.iloc[:, 1]
        pp_fy = data_2.max() - data_2.min()  # 峰峰值 = 最大值 - 最小值
        li.append(pp_fy)

        # 3.y方向切削力的“峰值”
        max_abs_fy = data_2.abs().max() # 峰值 = 绝对值.max()
        li.append(max_abs_fy)

        # 4.y方向切削力的“裕度因子”
        r_fy = (np.mean(np.sqrt(data_2.abs()))) ** 2  # 方根幅值
        m_fy = max_abs_fy / r_fy  # 裕度因子 = 峰值 / 方根幅值
        li.append(m_fy)

        # 5.y方向切削力的“波形因子”
        rms_fy = np.sqrt(np.mean(data_2.pow(2)))  # 均方根
        p_fy = data_2.abs().mean()  # 绝对平均幅值
        iw_fy = rms_fy / p_fy  # 波形因子 = 均方根 / 绝对平均幅值
        li.append(iw_fy)

        # 6.z方向切削力的“方根幅值”
        data_3 = data.iloc[:, 2]
        r_fz = (np.mean(np.sqrt(data_3.abs()))) ** 2
        li.append(r_fz)

        # 7.z方向切削力的“峰形因子”
        max_abs_fz = data_3.abs().max()  # 峰值
        rms_fz = np.sqrt(np.mean(data_3.pow(2)))  # 均方根
        ip_fz = max_abs_fz / rms_fz  # 峰形因子 =  峰值 / 均方根
        li.append(ip_fz)

        # 8.z方向切削力的“绝对平均幅值”
        p_fz = data_3.abs().mean()
        li.append(p_fz)

        # 9.z方向切削力的“脉冲因子”
        ii_fz = max_abs_fz / p_fz  # 脉冲因子 = 峰值 / 绝对平均幅值
        li.append(ii_fz)

        # 10.y方向的振动的“均方根值”
        data_5 = data.iloc[:, 4]
        rms_vy = np.sqrt(np.mean(data_5.pow(2)))
        li.append(rms_vy)

        return li


    def get_all_featrue(self, filename = r'./data/feature_c1.csv'):
        """
        遍历前n个CSV文件，提取每列的特征并保存到一个新的CSV文件中（n=315, 因为有315个文件）
        :param n: 文件数量
        """
        print('-------start output all feature------')
        n = 315  # c1、c2、c3分别有315个文件（表示不同磨损程度）

        # 得到一个二维数组： 行=》文件   列=》提取的特征
        all_feature = [self.get_file_feature(i) for i in range(1, n+1)]
        all_feature = np.array(all_feature).reshape(-1, 10)
        all_feature = pd.DataFrame(all_feature)

        # 保存
        all_feature.to_csv(filename, index=False)
        print('-------output all feature done------')


    def merge_csv(self, filename_feature='./data/feature_c1.csv', filename_rawy='./raw/c1_wear.csv', new_filename='./data/data_c1.csv'):
        """
        合并特征数据和目标变量（刀具磨损状态），生成最终的训练数据集。
        :param filename: 包含特征数据的CSV文件路径
        """
        data_feautre = pd.read_csv(filename_feature)  # 特征
        data_rawy = pd.read_csv(filename_rawy) # 还没处理的y

        # 当刀具的磨损量为0.165mm时，刀具达到失效状态
        # 三刃的话（有一个刃达到165即这之后都是失效状态）
        # （remaining use life）RUL定义为刀具从当前切削状态至失效状态所经历的时间（输出）
        # 由“剩余切削次数：失效序号-当前序号”与“总切削次数：失效序号+1”的比率来表征刀具的剩余使用寿命RUL
        data_wear = np.empty((315, 1))  # 创建一个空数组（装rul）

        row_wear = data_rawy.index[(data_rawy.iloc[:, 1:] > 165).any(axis=1)][0] # 失效开始的下标

        for i in range(row_wear):
            data_wear[i] = (row_wear - i - 1) / row_wear

        data_wear[row_wear:] = 0  # 没有寿命了


        # 列名为“rul”
        data_wear = pd.DataFrame(data_wear, columns=['rul'])

        # 按列拼接（添加上“rul”这一列）
        real_data = pd.concat((data_feautre, data_wear), axis=1)

        # 保存为csv文件
        real_data.to_csv(new_filename, index=False)

        print('---merge done---')

    def rename_col(self, filename='./data/data_c1.csv'):
        data = pd.read_csv(filename)

        data.columns = ['std_fx', 'pp_fy', 'max_abs_fy', 'm_fy', 'iw_fy', 'r_fz', 'ip_fz', 'p_fz', 'ii_fz', 'rms_vy', 'rul']

        data.to_csv(filename)

        print('修改列名成功')


if __name__ == '__main__':
    # 原始数据
    path1 = './raw/c1'
    path2 = './raw/c4'
    path3 = './raw/c6'

    # 提取特征之后数据集
    path1_save_feature = './data/feature_c1.csv'
    path2_save_feature = './data/feature_c4.csv'
    path3_save_feature = './data/feature_c6.csv'

    # 原始y数据集
    path1_raw_y = './raw/c1_wear.csv'
    path2_raw_y = './raw/c4_wear.csv'
    path3_raw_y = './raw/c6_wear.csv'

    # 合并“y”之后"最终数据集"
    path1_data = './data/data_c1.csv'
    path2_data = './data/data_c4.csv'
    path3_data = './data/data_c6.csv'

    # 1.处理c1数据集
    # dp_c1 = DataProcess(path1)
    # dp_c1.get_all_featrue(path1_save_feature)  # 提取特征保存到path1_save
    # dp_c1.merge_csv(path1_save_feature, path1_raw_y, path1_data)

    # dp_c1.rename_col(path1_data)  # 修改列名


    # 2.处理c4数据集
    # dp_c2 = DataProcess(path2)
    # dp_c2.get_all_featrue(path2_save_feature)  # 提取特征保存到path2_save
    # dp_c2.merge_csv(path2_save_feature, path2_raw_y, path2_data)

    # dp_c2.rename_col(path2_data)  # 修改列名

    # 3.处理c6数据集
    dp_c3 = DataProcess(path3)
    # dp_c3.get_all_featrue(path3_save_feature)  # 提取特征保存到path1_save
    # dp_c3.merge_csv(path3_save_feature, path3_raw_y, path3_data)

    # dp_c3.rename_col(path3_data)  # 修改列名



