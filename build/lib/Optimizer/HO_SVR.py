import numpy as np
import math
from scipy.special import gamma

def levy(n, m, beta):
    """
    Lévy飞行步长生成函数。
    用于增强算法的探索能力。
    参数：
        n: 种群数量
        m: 问题维度
        beta: Lévy分布形状参数（通常设为1.5）
    返回：
        z: 服从Lévy分布的步长矩阵（n x m）
    """

    # 计算标准差σ_w
    # gamma： 伽马分布
    num = gamma(1 + beta) * math.sin(math.pi * beta / 2)  # 分子
    den = gamma((1 + beta)/2) * beta * 2**((beta - 1)/2)  # 分母
    sigma_u = (num / den)**(1/beta)

    # 生成高斯分布的随机数 u 和 v
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    # Lévy公式计算步长
    z = u / (np.abs(v)**(1/beta))
    return z

def HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """
    河马优化算法主函数（Hippopotamus Optimization Algorithm, HO）
    模拟河马群体在自然界中的三种行为：觅食、防御捕食者、逃避捕食者。
    参数：
        SearchAgents: 种群数量
        Max_iterations: 最大迭代次数
        lowerbound: 搜索空间下界
        upperbound: 搜索空间上界
        dimension: 问题维度
        fitness: 目标适应度函数
    返回：
        Best_score: 全局最优适应度值
        Best_pos: 全局最优解位置
        HO_curve: 每代最优适应度记录
    """

    # 扩展边界为与种群维度一致的数组
    lowerbound = np.ones(dimension) * lowerbound
    upperbound = np.ones(dimension) * upperbound

    # 初始化种群位置
    X = np.zeros((SearchAgents, dimension))
    for i in range(dimension):
        X[:, i] = lowerbound[i] + np.random.rand(SearchAgents) * (upperbound[i] - lowerbound[i])

    # 计算每个个体的适应度值
    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]          # 当前个体的位置
        fit[i] = fitness(L)  # 调用目标函数计算适应度

    # 存储每一代的最优适应度值
    best_so_far = np.zeros(Max_iterations)

    # 主循环开始
    for t in range(Max_iterations):

        # 更新当前代中最优个体
        best = np.min(fit)            # 当前代最优适应度
        location = np.argmin(fit)     # 当前代最优个体索引

        # 如果是第一次迭代或找到了更优解，则更新全局最优
        if t == 0:
            Xbest = X[location, :]
            fbest = best
        elif best < fbest:
            fbest = best
            Xbest = X[location, :]

        # Phase 1: 河马觅食阶段（探索 Exploration）
        for i in range(int(SearchAgents / 2)):

            Dominant_hippopotamus = Xbest  # 当前全局最优作为“主导河马”
            I1 = np.random.randint(1, 3)   # 随机选择1或2
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, size=2)  # 二元随机数(都是0或1)

            # 随机选择一组河马用于平均
            # np.random.randint: 1到SearchAgents（包含）之间的整数
            RandGroupNumber = np.random.randint(1, SearchAgents + 1)
            # np.random.choice： 来随机选择指定数量(RandGroupNumber)的个体索引
            RandGroup = np.random.choice(SearchAgents, RandGroupNumber, replace=False)

            # 计算该组河马的平均位置（平均参数mg1）
            if len(RandGroup) != 1:
                MeanGroup = np.mean(X[RandGroup, :], axis=0)
            else:
                MeanGroup = X[RandGroup[0], :]

            # 定义不同策略下的参数集合 Alfa （h）
            Alfa = {
                1: I2 * np.random.rand(dimension) + (1 - Ip1[0]),
                2: 2 * np.random.rand(dimension) - 1,
                3: np.random.rand(dimension),
                4: I1 * np.random.rand(dimension) + (1 - Ip1[1]),
                5: np.random.rand()
            }

            A = Alfa[np.random.randint(1, 6)]  # 随机选择两个策略参数
            B = Alfa[np.random.randint(1, 6)]

            # 策略1：基于主导河马进行位置更新
            X_P1 = X[i, :] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i, :])

            T = math.exp(-t / Max_iterations)  # 温度因子，控制开发与探索比例

            # 策略2：雌性/幼崽河马进行位置更新
            if T > 0.6:
                X_P2 = X[i, :] + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i, :] + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = (upperbound - lowerbound) * np.random.rand(dimension) + lowerbound

            # 限制新解在搜索范围内
            X_P1 = np.clip(X_P1, lowerbound, upperbound)
            X_P2 = np.clip(X_P2, lowerbound, upperbound)

            # 判断是否接受新解
            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

            L2 = X_P2
            F_P2 = fitness(L2)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        # Phase 2: 面对捕食者的策略（探索 Exploration）
        for i in range(int(SearchAgents / 2), SearchAgents):

            # 随机生成一个“捕食者”位置
            predator = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)

            L = predator
            F_HL = fitness(L)  # 捕食者位置的适应度

            # 计算当前个体与捕食者的距离(D)
            distance2Leader = np.abs(predator - X[i, :])

            # 设置控制参数(随机浮点数)
            b = np.random.uniform(2, 4) # f
            c = np.random.uniform(1, 1.5) # c
            d = np.random.uniform(2, 3) # d
            l = np.random.uniform(-2 * math.pi, 2 * math.pi) # 2pi*g
            RL = 0.05 * levy(SearchAgents, dimension, 1.5)  # 使用Lévy飞行步长

            # 根据适应度决定靠近还是远离捕食者
            if fit[i] > F_HL:  # 捕食者巨靠近
                X_P3 = RL[i, :] * predator + (b / (c - d * math.cos(l))) * (1 / distance2Leader)
            else:  # 捕食者离的还比较远
                X_P3 = RL[i, :] * predator + (b / (c - d * math.cos(l))) * (1 / (2 * distance2Leader + np.random.rand(dimension)))

            # 限制范围
            X_P3 = np.clip(X_P3, lowerbound, upperbound)

            # 判断是否接受新解
            L = X_P3
            F_P3 = fitness(L)
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3

        # Phase 3: 逃避捕食者（开发 Exploitation）
        for i in range(SearchAgents):

            # 随着迭代增加，局部搜索范围逐渐缩小
            LO_LOCAL = lowerbound / (t + 1)
            HI_LOCAL = upperbound / (t + 1)

            # 不同策略下的扰动因子（字典）
            Alfa = {
                1: 2 * np.random.rand(dimension) - 1,
                2: np.random.rand(),
                3: np.random.randn()
            }

            D = Alfa[np.random.randint(1, 4)]  # 随机选择一个扰动方式
            X_P4 = X[i, :] + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))

            # 限制范围
            X_P4 = np.clip(X_P4, lowerbound, upperbound)

            # 判断是否接受新解
            L = X_P4
            F_P4 = fitness(L)
            if F_P4 < fit[i]:
                X[i, :] = X_P4
                fit[i] = F_P4

        # 记录本次迭代的最优适应度（一开始就取了最优的情况）
        best_so_far[t] = fbest
        # print(f'Iteration {t+1}: Best Cost = {best_so_far[t]}')

    # 返回最终结果
    Best_score = fbest # 最优参数的适应度值
    Best_pos = Xbest  # 最优参数
    HO_curve = best_so_far # 每一轮最优参数情况

    return Best_pos # 把参数返回去就行


