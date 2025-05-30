from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from sklearn import metrics
import csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import random
import copy


# 染色体个体
class GAIndividual:

    def __init__(self, vardim, bound, fitness_function):
        '''
        vardim: 参数个数2（C，gamma）
        bound: 边界（这个是粒子群算法也是要设置...感觉最难！！！）
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness_function = fitness_function
        self.fitness = 0.  # 存当前“染色体个体”的适应度值

    def generate(self):
        '''
        产生随机参数C，gamma
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):  # 要在边界范围内
            self.chrom[i] = self.bound[0, i] + \
                            (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        根据参数==》计算适应度函数
        '''
        self.fitness = self.fitness_function(self.chrom)

class GeneticAlgorithm:

    def __init__(self, sizepop, MAXGEN, bound, vardim, fitness_function, params):

        self.sizepop = sizepop # 种群的个体个数（染色体的个数）
        self.MAXGEN = MAXGEN  # 最大轮数
        self.bound = bound  # 变量边界
        self.vardim = vardim # 参数个数（2）
        self.fitness_function = fitness_function # 适应度函数
        self.params = params  # 交叉率，变异率，α参数

        self.population = [] # 存放当前种群中的染色体个体
        self.fitness = np.zeros((self.sizepop, 1))  # 种群的每一个染色体个体的适应度值
        self.trace = np.zeros((self.MAXGEN, 3)) # 每一轮的（目前的最优适应度, 当前轮平均适应度， 当前轮的最大适应度）


    def initialize(self):
        '''
        为每一个个体创建一个“染色体对象”
        '''
        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound, self.fitness_function)
            ind.generate() # 生成参数C，gamma
            self.population.append(ind)

    def evaluate(self):
        '''
        求当前群体的每一个染色体的“适应度”，记录到fitness数组
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        '''
        算法主体
        '''
        self.t = 0  # 当前轮数
        self.initialize()  # 初始化
        self.evaluate() # 计算对应的适应度值
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)  # 取最大适应度的下标
        self.best = copy.deepcopy(self.population[bestIndex]) # 复制出最大适应度的“染色体个体”
        self.avefitness = np.mean(self.fitness) # 当前轮的平均适应度
        self.maxfitness = np.max(self.fitness) # 当前轮的最大适应度

        self.trace[self.t, 0] = self.best.fitness  # 目前的最优适应度
        self.trace[self.t, 1] = self.avefitness  # 当前轮的平均适应度
        self.trace[self.t, 2] = self.maxfitness  # 当前轮的最大适应度

        # 将后面轮也跑完...
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.selectionOperation()  # 选择染色体个体
            self.crossoverOperation() # 交叉
            self.mutationOperation() # 变异

            # 同理（开始计算当前种群的适应度值，取最大的适应度值）
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:  # 看一下是否能够超过我的原先最大适应度值
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)  # 当前轮的平均适应度
            self.maxfitness = np.max(self.fitness)  # 当前轮的最大适应度

            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            self.trace[self.t, 2] = self.maxfitness

        # 返回最优参数即可
        return self.best.chrom

    # 轮盘赌选择法（Roulette Wheel Selection）来选择个体
    # 这种方法确保了适应度越高的个体被选择的机会越大，但同时也给予低适应度个体一定的生存机会
    def selectionOperation(self):
        newpop = []  # 创建一个空列表用于存储新种群
        totalFitness = np.sum(self.fitness) # 计算当前种群所有个体适应度值的总和
        accuFitness = np.zeros((self.sizepop, 1)) # 创建一个与种群大小相同的数组，用于存储累积概率

        sum1 = 0.
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness # # 当前个体的累积概率为其前面所有个体的概率加上自身概率
            sum1 = accuFitness[i]

        # 根据轮盘赌选择法选择个体，创建新种群
        for i in range(0, self.sizepop):
            r = random.random()  # 随机一个概率
            idx = 0
            for j in range(0, self.sizepop - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop

    # 线性组合交叉（Blend Crossover）
    def crossoverOperation(self):
        newpop = []  # 创建一个空列表，用于存储经过交叉操作后的新种群

        #  # 每次处理两个个体进行交叉（种群大小为10--偶数）
        for i in range(0, self.sizepop, 2):
            # 随机选择两个不同的个体索引作为父代
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)

            # 确保两个父代个体不同（避免自交）
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)

            # 将这两个父代个体深拷贝到新种群中，准备进行交叉操作
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))

            # 生成一个随机数 r，判断是否执行交叉操作
            r = random.random()
            if r < self.params[0]: # params[0]:交叉率
                # 随机选择一个交叉起始位置（从该位置开始交换基因）
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    # params[2]：α参数
                    # 线性组合交叉公式： 可以产生两个新的中间解，有助于保持种群多样性
                    newpop[i].chrom[j] = newpop[i].chrom[
                                             j] * self.params[2] + (1 - self.params[2]) * newpop[i + 1].chrom[j]
                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                                             (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop

    # 变异操作（Mutation）
    # 它通过随机扰动个体的某些基因位来增加种群的多样性，防止算法陷入局部最优
    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = [] # 创建一个空列表用于存储经过变异操作后的新种群

        for i in range(0, self.sizepop):
            # 深拷贝当前个体，避免直接修改原始种群
            newpop.append(copy.deepcopy(self.population[i]))

            # 生成一个 [0,1) 之间的随机数，决定是否对该个体进行变异
            r = random.random()
            if r < self.params[1]:  # params[1]：变异率
                # 随机选择一个基因位置进行变异
                mutatePos = random.randint(0, self.vardim - 1)
                # 再次使用随机数 theta 来决定变异方向（向上或向下）
                theta = random.random()
                if theta > 0.5: # 向上
                    # 公式含义：new_gene = current_gene - (current_gene - lower_bound) * factor
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                                                     mutatePos] - (
                                                             newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (
                                                             1 - random.random() ** (1 - self.t / self.MAXGEN))
                else: # 向下
                    # 公式含义：new_gene = current_gene + (upper_bound - current_gene) * factor
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                                                     mutatePos] + (
                                                             self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (
                                                             1 - random.random() ** (1 - self.t / self.MAXGEN))
        self.population = newpop
