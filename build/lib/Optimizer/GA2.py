import numpy as np
from deap import base, creator, tools, algorithms


def GA(n_chrom, n_iterations, bounds, dimension, fitness_function, options):

    # 1. 定义遗传算法问题
    creator.create("FitnessMse", base.Fitness, weights=(-1.0,))  # -1表示最小化该目标
    creator.create("Individual", list, fitness=creator.FitnessMse) # 个体类(个体的基因存储在一个列表中,使用该适应度函数)

    # 2. 初始化工具箱
    # 初始化工具箱
    toolbox = base.Toolbox()
    # 注册属性生成器(C, gamma)
    toolbox.register("attr_C", np.random.uniform, bounds[0, 0], bounds[1, 0])
    toolbox.register("attr_gamma", np.random.uniform, bounds[0, 1], bounds[1, 1])
    # 定义个体生成器
    # n=1 即每个个体只包含一对 C 和 gamma 值
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_C, toolbox.attr_gamma), n=1)
    # 定义种群生成器
    # tools.initRepeat 用于生成种群
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 3. 配置遗传操作
    # 混合交叉  A,B为父代
    # (alpha: 子代的基因将从 (A - alpha * (B-A)) 到 (B + alpha * (B-A)) 范围内随机选取)
    toolbox.register("mate", tools.cxBlend, alpha=options[2])
    # 注册变异操作
    # mu=0 和 sigma=1 分别表示高斯分布的均值和标准差
    # indpb=0.2 表示每个基因独立变异的概率为 20%
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    # 注册选择操作
    # tools.selTournament 是锦标赛选择法
    # tournsize=3 意味着每次从 3 个个体中选择最好的一个进入下一代
    toolbox.register("select", tools.selTournament, tournsize=3)
    # 注册评估函数
    toolbox.register("evaluate", fitness_function)

    # 4. 创建初始种群
    population = toolbox.population(n=n_chrom)

    # 5. 评估初始种群
    fitnesses = list(map(toolbox.evaluate, population))  # 对每一个个体计算适应度函数
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit  # 将适应度值 fit 赋予个体 ind 的 fitness 属性 (要幅值元组)

    # 6. 运行遗传算法
    # cxpb: 交叉率
    # mutpb: 编译率
    # n_gen: 迭代数
    # verbose: 是否打印进化过程(默认为True)
    algorithms.eaSimple(population, toolbox, cxpb=options[0], mutpb=options[1], ngen=n_iterations,
                        verbose=False)

    # 7.获取最佳参数
    best_ind = tools.selBest(population, k=1)[0]
    best_C = best_ind[0]
    best_gamma = best_ind[1]
    return best_C, best_gamma