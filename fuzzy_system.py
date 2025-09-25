import random
import numpy as np
import operator
import math

NP = 50
seita = 10 ** (-10)


# 给当前任务的种群，计算迁移后代存活率
def get_survive(population):
    total = 0
    survive = 0
    # 排序后的数组，2NP中统计所有迁移后代，在前NP中的是可以保留到下一代的
    for i in range(2 * NP):
        if population[i]['child'] == True:
            total += 1
            if i < NP:
                survive += 1
    # 如果没有迁移后代，迁移后代存活率为0
    if total == 0:
        return 0
    return survive / total


# 给当前任务的种群，和最优解，计算迁移后代距离最优解的距离
def get_d(population, t1_best_X):
    d = 0
    cnt = 0
    # 排序后的数组，在前NP中的是可以保留到下一代的
    for i in range(NP):
        if population[i]['child'] == True:
            f1 = np.sum((population[i]['information'] - t1_best_X) ** 2)
            f2 = np.sqrt(f1)
            d = d + f2
            cnt += 1
    # 前NP个中，没有迁移后代，给一个很大的数字
    if cnt == 0:
        d = 10
    else:
        d = d / cnt
    return d


# 给当前种群计算迁移后代平均改变率
def get_r(population):
    r = 0
    cnt = 0
    # 排序后的数组，在前NP中的是可以保留到下一代的
    for i in range(NP):
        if population[i]['child'] == True:
            r = r + (population[i]['cost'] + seita) / (population[i]['parent_cost'] + seita)
            cnt += 1
    # 前NP个中，没有迁移后代，给一个很大的数字
    if cnt == 0:
        r = 10
    else:
        r = r / cnt
    return r


# 迁移后代存活率隶属度
def survive_u(survive):
    if survive >=0.75:
        u0 = 0
        u1 = 1
    else:
        # 存活率低隶属度-实线
        u0 = 1 -4.0/3*survive
        # 存活率高隶属度-虚线
        u1 = 4.0/3 * survive
    return np.array([u0, u1])


# 适应值改变率隶属度
def r_u(r):
    if 0.9 <= r and r <= 1:
        # 比率小-变化大-隶属度-实线
        u0 = -10 * r + 10
        # 比率大-变化小-隶属度-虚线
        u1 = -9 + 10 * r
    elif r > 1:
        u0 = 0
        u1 = 1
    elif r < 0.9:
        u0 = 1
        u1 = 0
    return np.array([u0, u1])


# 距离隶属度

def d_u(d):
    if d <= 0.0001:
        # 距离小隶属度-实线
        u0 = 1 - 10000*d
        # 距离大隶属度-虚线
        u1 = 10000 * d
    else:
        u0 = 0
        u1 = 1
    return np.array([u0, u1])


# rmp的设定
def RMP(i, j, k):
    if i == 0:
        if j == 0:
            if k == 0:
                return 0.7
            else:
                return 0.3
        elif j == 1:
            if k == 0:
                return 0.3
            else:
                return 0.1
    elif i == 1:
        if j == 0:
            if k == 0:
                return 1.0
            else:
                return 0.7
        elif j == 1:
            if k == 0:
                return 0.7
            else:
                return 0.3


# 计算rmp
def get_rmp(U):
    tmp1 = 0
    tmp2 = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                tmp1 = tmp1 + U[0][i] * U[1][j] * U[2][k] * RMP(i, j, k)
                tmp2 = tmp2 + U[0][i] * U[1][j] * U[2][k]
    return tmp1 / tmp2
