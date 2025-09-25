import random
import numpy as np
import WCCI2020MTSO as WCCI
import CEC2017MTSO as CEC
import operator

# 此为一般种群操作
NP = 50  # 个体数目
maxD = 50  # 所有问题的最大维度
# GA
lameda = 0.25  # 算术交叉的参数
oumeiga = 0.25  # 几何交叉的参数
aerfa = 0.3  # BLX-alpha的参数
mu = 10  # SBX交叉的参数
mum = 5  # 多项式变异的参数
# DE
F = 0.5  # 缩放因子
Cr = 0.6  # 交叉概率


# 多种群初始化，随机生成NP个个体，每个个体的维数为所有任务中最大的那个
def t_init():
    pop1 = []
    for i in range(NP):
        tmp = {}
        tmp["information"] = np.random.uniform(low=0, high=1, size=[maxD])
        tmp["child"] = False  # 标记是否为迁移后代，True为迁移后代
        pop1.append(tmp)
    pop2 = []
    for i in range(NP):
        tmp = {}
        tmp["information"] = np.random.uniform(low=0, high=1, size=[maxD])
        tmp["child"] = False  # 标记是否为迁移后代，True为迁移后代
        pop2.append(tmp)
    return pop1, pop2


# 对于初始多种群population，就task问题，进行初步评价,计算适应值，返回最新种群信息和最优信息
def t_evaluate(population, task):
    t1_gbest_F = np.inf  # 任务的最优值
    t1_gbest_X = []  # 任务的最优个体
    # 评价因子代价cost
    for i in range(NP):
        population[i]["cost"] = task.function(population[i]["information"])
        if t1_gbest_F > population[i]["cost"]:  # 更新任务1最优
            t1_gbest_F = population[i]["cost"]
            t1_gbest_X = population[i]["information"].copy()
    return population, t1_gbest_F, t1_gbest_X



# population种群中的i,j个体以k方式交叉,其中6为模拟二进制交叉SBX（多种群）
def t_crossover(population1,population2, i, j):
    c1 = []  # 新个体1的information
    c2 = []  # 新个体2的information

    u = np.random.uniform(low=0, high=1, size=[maxD])
    cf = np.zeros(maxD)
    for x in range(maxD):
        if u[x] <= 0.5:
            cf[x] = (2 * u[x]) ** (1 / (mu + 1))
        else:
            cf[x] = (2 * (1 - u[x])) ** (-1 / (mu + 1))
        tmp = 0.5 * ((1 - cf[x]) * population1[i]["information"][x] + (1 + cf[x]) * population2[j]["information"][x])
        c1.append(tmp)
        tmp = 0.5 * ((1 + cf[x]) * population1[i]["information"][x] + (1 - cf[x]) * population2[j]["information"][x])
        c2.append(tmp)
    c1 = np.array(c1)  # 列表转nparray
    c2 = np.array(c2)  # 列表转nparray
    # 控边界
    # 将大于1的元素设为1，小于0的元素设为0
    c1[c1 > 1] = 1
    c1[c1 < 0] = 0
    c2[c2 > 1] = 1
    c2[c2 < 0] = 0
    d1 = {}  # 新个体1的整体信息
    d2 = {}  # 新个体2的整体信息
    d1["information"] = c1
    d2["information"] = c2
    return d1, d2

# array1个体多项式变异(多种群)
def t_mutated(array1):
    c = []
    d = {}
    for x in range(maxD):
        rand = random.uniform(0, 1)
        if rand < 1 / maxD:  # 以1/maxD的概率变异
            u = random.uniform(0, 1)
            if u <= 0.5:  # 以第一个公式变异
                tmp1 = (2 * u) ** (1 / (1 + mum)) - 1
                tmp2 = array1["information"][x] + tmp1 * array1["information"][x]
                c.append(tmp2)
            else:  # 以第二个公式变异
                tmp1 = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
                tmp2 = array1["information"][x] + tmp1 * (1 - array1["information"][x])
                c.append(tmp2)
        else:  # 不变异
            c.append(array1["information"][x])
    c = np.array(c)
    # 控边界
    c[c > 1] = 1
    c[c < 0] = 0
    d["information"] = c
    return d


# 多种群DE/rand/1
def t_DE(array1, array2, array3, array4):
    # 执行变异操作
    v = array1['information'] + F * (array2['information'] - array3['information'])
    # 保证解的有效性
    v[v < 0] = 0
    v[v > 1] = 1
    # 交叉操作
    krand = random.randint(0, maxD - 1)  # jrand
    rand = np.random.uniform(low=0, high=1, size=[maxD])  # rand
    u = array4['information'].copy()
    for k in range(maxD):
        if rand[k] <= Cr or k == krand:
            u[k] = v[k]
    d = {}
    d['information'] = u.copy()
    d['child'] = False
    return d
