import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import pyswarms as ps

# 优化函数
def optimize_svm(n_particles, n_iterations, bounds, dimension, fitness_function, options):

    def _fitness_function(params):  # params: 传入的多个粒子位置(参数元组--（C,gamma）)
        fitness_values = []
        for p in params:  # 根据粒子位置，计算当前的“适应值” （以用来不断的最小化）
            fitness = fitness_function(p)
            fitness_values.append(fitness)
        return fitness_values

    # 1.初始化
    cost_history = np.zeros(n_iterations)

    # 2.1 构建优化器PSO
    # n_particles： 粒子个数
    # dimensions： 要优化的参数个数2（C，gamma）
    # bounds:  参数的边界（C，gamma）        //这个怎么确定呢？？？
    # options: 包含控制优化过程的超参数，如惯性权重w、个体学习因子c1、群体学习因子c2等
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimension, bounds=bounds, options=options)

    # 2.2 优化器PSO开始优化参数
    # _fitness_function: 所有粒子所在位置的“适应值”计算
    # iters： 迭代次数
    best_cost, best_params = optimizer.optimize(_fitness_function, iters=n_iterations)

    # 在每次迭代保存代价值
    for i, cost in enumerate(optimizer.cost_history):
        cost_history[i] = cost

    # 返回最优参数即可
    return best_params