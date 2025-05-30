#coding=gbk

import pandas as pd
import numpy as np
import os

class DataProcess:
    def __init__(self, path):
        """
        ��ʼ�����������������ļ�·��������¼��ʼʱ�䡣
        :param path: �����ļ���·��
        """
        self.path = path  # �洢�����ļ�·��

    def get_all_file(self):
        """
        ��ȡ�ļ����е������ļ�����
        :return: �ļ��б�
        """
        files = os.listdir(self.path)
        # files.sort(key=lambda x: int(x[:-4]))  # ���ļ�������ȥ������λ��
        s = []
        for file in files:
            if not os.path.isdir(os.path.join(self.path, file)):  # �ж��Ƿ�Ϊ�ļ�
                filename = os.path.join(self.path, file)  # ƴ����·��
                s.append(filename)
        return s

    def get_data(self, i):
        """
        ����"����"��ȡ��Ӧ��CSV�ļ����ݡ�
        :param i: �ļ�����
        :return: DataFrame��ʽ������
        """
        list_files = self.get_all_file()
        data = pd.read_csv(list_files[i-1])
        return data

    def get_shape(self, n=2):
        """
        ��ȡǰn��csv�ļ�����״������������������ӡ��
        :param n: �ļ�����
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
        ���ǰn��csv�ļ��������С��ƽ��������
        :param n: �ļ�����
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
        ����ǰn��csv�ļ������������ܺͲ���ӡ��
        :param n: �ļ�����
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
        ��ȡ����CSV�ļ���10��ͳ��������
        ����csv�ļ���ÿһ�зֱ��Ӧx��y��z������������ x��y��z������𶯣� AE-RMS (V)
        :param i: �ļ�����
        :return: �����б�
        """

        data = self.get_data(i)  # ȡ����i��csv�ļ�

        # �������
        li = []

        # 1.x�������������ġ���׼�
        data_1 = data.iloc[:, 0]
        std_fx = data_1.std()  # ��׼��
        li.append(std_fx)

        # 2.y�����������ġ����ֵ��
        data_2 = data.iloc[:, 1]
        pp_fy = data_2.max() - data_2.min()  # ���ֵ = ���ֵ - ��Сֵ
        li.append(pp_fy)

        # 3.y�����������ġ���ֵ��
        max_abs_fy = data_2.abs().max() # ��ֵ = ����ֵ.max()
        li.append(max_abs_fy)

        # 4.y�����������ġ�ԣ�����ӡ�
        r_fy = (np.mean(np.sqrt(data_2.abs()))) ** 2  # ������ֵ
        m_fy = max_abs_fy / r_fy  # ԣ������ = ��ֵ / ������ֵ
        li.append(m_fy)

        # 5.y�����������ġ��������ӡ�
        rms_fy = np.sqrt(np.mean(data_2.pow(2)))  # ������
        p_fy = data_2.abs().mean()  # ����ƽ����ֵ
        iw_fy = rms_fy / p_fy  # �������� = ������ / ����ƽ����ֵ
        li.append(iw_fy)

        # 6.z�����������ġ�������ֵ��
        data_3 = data.iloc[:, 2]
        r_fz = (np.mean(np.sqrt(data_3.abs()))) ** 2
        li.append(r_fz)

        # 7.z�����������ġ��������ӡ�
        max_abs_fz = data_3.abs().max()  # ��ֵ
        rms_fz = np.sqrt(np.mean(data_3.pow(2)))  # ������
        ip_fz = max_abs_fz / rms_fz  # �������� =  ��ֵ / ������
        li.append(ip_fz)

        # 8.z�����������ġ�����ƽ����ֵ��
        p_fz = data_3.abs().mean()
        li.append(p_fz)

        # 9.z�����������ġ��������ӡ�
        ii_fz = max_abs_fz / p_fz  # �������� = ��ֵ / ����ƽ����ֵ
        li.append(ii_fz)

        # 10.y������񶯵ġ�������ֵ��
        data_5 = data.iloc[:, 4]
        rms_vy = np.sqrt(np.mean(data_5.pow(2)))
        li.append(rms_vy)

        return li


    def get_all_featrue(self, filename = r'./data/feature_c1.csv'):
        """
        ����ǰn��CSV�ļ�����ȡÿ�е����������浽һ���µ�CSV�ļ��У�n=315, ��Ϊ��315���ļ���
        :param n: �ļ�����
        """
        print('-------start output all feature------')
        n = 315  # c1��c2��c3�ֱ���315���ļ�����ʾ��ͬĥ��̶ȣ�

        # �õ�һ����ά���飺 ��=���ļ�   ��=����ȡ������
        all_feature = [self.get_file_feature(i) for i in range(1, n+1)]
        all_feature = np.array(all_feature).reshape(-1, 10)
        all_feature = pd.DataFrame(all_feature)

        # ����
        all_feature.to_csv(filename, index=False)
        print('-------output all feature done------')


    def merge_csv(self, filename_feature='./data/feature_c1.csv', filename_rawy='./raw/c1_wear.csv', new_filename='./data/data_c1.csv'):
        """
        �ϲ��������ݺ�Ŀ�����������ĥ��״̬�����������յ�ѵ�����ݼ���
        :param filename: �����������ݵ�CSV�ļ�·��
        """
        data_feautre = pd.read_csv(filename_feature)  # ����
        data_rawy = pd.read_csv(filename_rawy) # ��û�����y

        # �����ߵ�ĥ����Ϊ0.165mmʱ�����ߴﵽʧЧ״̬
        # ���еĻ�����һ���дﵽ165����֮����ʧЧ״̬��
        # ��remaining use life��RUL����Ϊ���ߴӵ�ǰ����״̬��ʧЧ״̬��������ʱ�䣨�����
        # �ɡ�ʣ������������ʧЧ���-��ǰ��š��롰������������ʧЧ���+1���ı������������ߵ�ʣ��ʹ������RUL
        data_wear = np.empty((315, 1))  # ����һ�������飨װrul��

        row_wear = data_rawy.index[(data_rawy.iloc[:, 1:] > 165).any(axis=1)][0] # ʧЧ��ʼ���±�

        for i in range(row_wear):
            data_wear[i] = (row_wear - i - 1) / row_wear

        data_wear[row_wear:] = 0  # û��������


        # ����Ϊ��rul��
        data_wear = pd.DataFrame(data_wear, columns=['rul'])

        # ����ƴ�ӣ�����ϡ�rul����һ�У�
        real_data = pd.concat((data_feautre, data_wear), axis=1)

        # ����Ϊcsv�ļ�
        real_data.to_csv(new_filename, index=False)

        print('---merge done---')

    def rename_col(self, filename='./data/data_c1.csv'):
        data = pd.read_csv(filename)

        data.columns = ['std_fx', 'pp_fy', 'max_abs_fy', 'm_fy', 'iw_fy', 'r_fz', 'ip_fz', 'p_fz', 'ii_fz', 'rms_vy', 'rul']

        data.to_csv(filename)

        print('�޸������ɹ�')


if __name__ == '__main__':
    # ԭʼ����
    path1 = './raw/c1'
    path2 = './raw/c4'
    path3 = './raw/c6'

    # ��ȡ����֮�����ݼ�
    path1_save_feature = './data/feature_c1.csv'
    path2_save_feature = './data/feature_c4.csv'
    path3_save_feature = './data/feature_c6.csv'

    # ԭʼy���ݼ�
    path1_raw_y = './raw/c1_wear.csv'
    path2_raw_y = './raw/c4_wear.csv'
    path3_raw_y = './raw/c6_wear.csv'

    # �ϲ���y��֮��"�������ݼ�"
    path1_data = './data/data_c1.csv'
    path2_data = './data/data_c4.csv'
    path3_data = './data/data_c6.csv'

    # 1.����c1���ݼ�
    # dp_c1 = DataProcess(path1)
    # dp_c1.get_all_featrue(path1_save_feature)  # ��ȡ�������浽path1_save
    # dp_c1.merge_csv(path1_save_feature, path1_raw_y, path1_data)

    # dp_c1.rename_col(path1_data)  # �޸�����


    # 2.����c4���ݼ�
    # dp_c2 = DataProcess(path2)
    # dp_c2.get_all_featrue(path2_save_feature)  # ��ȡ�������浽path2_save
    # dp_c2.merge_csv(path2_save_feature, path2_raw_y, path2_data)

    # dp_c2.rename_col(path2_data)  # �޸�����

    # 3.����c6���ݼ�
    dp_c3 = DataProcess(path3)
    # dp_c3.get_all_featrue(path3_save_feature)  # ��ȡ�������浽path1_save
    # dp_c3.merge_csv(path3_save_feature, path3_raw_y, path3_data)

    # dp_c3.rename_col(path3_data)  # �޸�����



