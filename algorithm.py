import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import WCCI2020MTSO as WCCI
import CEC2017MTSO as CEC
import population_operation_tmp as po
import operator
import fuzzy_system as fs

G = 1000  # 迭代次数
k = 6  # 交叉方式选择SBX交叉
NP = 50  # 个体数目
maxD = 50  # 所有问题的最大维度
eop = 0.9  # 进化算子的选择概率
f=0.5  # rmp的改变率

# 算法本身
def algorithm(task1, task2, question):
    # 初始化种群
    population1, population2 = po.t_init()
    # 对种群个体进行初始评价
    population1, t1_gbest_F, t1_gbest_X = po.t_evaluate(population1, task1)
    population2, t2_gbest_F, t2_gbest_X = po.t_evaluate(population2, task2)
    t1 = []  # 记录任务一每一代最优
    t2 = []  # 记录任务二每一代最优
    Rmp1 = []  # 记录任务一每一代rmp
    Rmp2 = []  # 记录任务二每一代rmp
    rmp1 = 0.3  # 任务1的rmp
    rmp2 = 0.3  # 任务2的rmp
    Rmp1.append(rmp1)
    Rmp2.append(rmp2)

    # 进行迭代
    for g in range(G):
        # 遍历种群1
        for i in range(NP):
            # 种群1以rmp1的概率知识迁移
            rand = random.uniform(0, 1)
            if rand < rmp1:
                r1 = random.randint(0, NP - 1)
                # 随机选择种群2里的一个个体进行SBX交叉
                d1, d2 = po.t_crossover(population1, population2, i, r1)
                # 交叉了还是要变异的
                d1 = po.t_mutated(d1)
                d2 = po.t_mutated(d2)
                # 垂直文化传播，随机选择一个加入种群1
                if random.uniform(0, 1) < 0.5:
                    d1['cost'] = task1.function(d1["information"])
                    d1['child'] = True
                    d1['parent_cost'] = population1[i]['cost']
                    population1.append(d1)
                else:
                    d2['cost'] = task1.function(d2["information"])
                    d2['child'] = True
                    d2['parent_cost'] = population1[i]['cost']
                    population1.append(d2)
            # 种群1进行种群内进化
            else:
                # 以eop的概率进行DE/rand/1
                if random.uniform(0, 1) < eop:
                    indexs = [index for index in range(NP) if index != i]
                    r1, r2, r3 = np.random.choice(indexs, 3, replace=False)  # 选3个不相同的下标
                    d = po.t_DE(population1[r1], population1[r2], population1[r3], population1[i])
                    d['cost'] = task1.function(d["information"])
                    population1.append(d)
                # 以1-eop的概率进行SBX+多项式变异
                else:
                    r1 = random.randint(0, NP - 1)
                    d1, d2 = po.t_crossover(population1, population1, i, r1)
                    # 交叉了还是要变异的
                    d1 = po.t_mutated(d1)
                    d1['child'] = False
                    d1['cost'] = task1.function(d1["information"])
                    population1.append(d1)

        # 遍历种群2
        for i in range(NP):
            # 种群2以rmp2的概率知识迁移
            rand = random.uniform(0, 1)
            if rand < rmp2:
                # 随机选择种群1里的一个个体进行SBX交叉
                r1 = random.randint(0, NP - 1)
                d1, d2 = po.t_crossover(population2, population1, i, r1)
                # 交叉了还是要变异的
                d1 = po.t_mutated(d1)
                d2 = po.t_mutated(d2)
                # 垂直文化传播，随机选择一个加入种群1
                if random.uniform(0, 1) < 0.5:
                    d1['cost'] = task2.function(d1["information"])
                    d1['child'] = True
                    d1['parent_cost'] = population2[i]['cost']
                    population2.append(d1)
                else:
                    d2['cost'] = task2.function(d2["information"])
                    d2['child'] = True
                    d2['parent_cost'] = population2[i]['cost']
                    population2.append(d2)
            # 种群2进行种群内进化
            else:
                # 以eop的概率进行DE/rand/1
                if random.uniform(0, 1) < eop:
                    indexs = [index for index in range(NP) if index != i]
                    r1, r2, r3 = np.random.choice(indexs, 3, replace=False)  # 选3个不相同的下标
                    d = po.t_DE(population2[r1], population2[r2], population2[r3], population2[i])
                    d['cost'] = task2.function(d["information"])
                    population2.append(d)
                # 以1-eop的概率进行SBX+多项式变异
                else:
                    r1 = random.randint(0, NP - 1)
                    d1, d2 = po.t_crossover(population2, population2, i, r1)
                    # 交叉了还是要变异的
                    d1 = po.t_mutated(d1)
                    d1['child'] = False
                    d1['cost'] = task2.function(d1["information"])
                    population2.append(d1)

        # 当代个体生成完毕，且适应值求解完成，开始评价
        # 根据因子代价1排序
        population1 = sorted(population1, key=operator.itemgetter('cost'))
        population2 = sorted(population2, key=operator.itemgetter('cost'))
        # 模糊评价
        # 种群1计算rmp1
        f1 = fs.get_survive(population1)
        d1 = fs.get_d(population1, t1_gbest_X)
        r1 = fs.get_r(population1)



        uf = fs.survive_u(f1)
        ud = fs.d_u(d1)
        ur = fs.r_u(r1)
        U = []
        U.append(uf)
        U.append(ud)
        U.append(ur)
        U = np.array(U)
        rmp1_tmp = fs.get_rmp(U)


        # if f1 == 0:
        #     rmp1_tmp = rmp1
        # else:
        #     uf = fs.survive_u(f1)
        #     ud = fs.d_u(d1)
        #     ur = fs.r_u(r1)
        #     U = []
        #     U.append(uf)
        #     U.append(ud)
        #     U.append(ur)
        #     U = np.array(U)
        #     rmp1_tmp = fs.get_rmp(U)

        # print(U,rmp1)
        # 种群2计算rmp2
        f2 = fs.get_survive(population2)
        d2 = fs.get_d(population2, t2_gbest_X)
        r2 = fs.get_r(population2)



        uf = fs.survive_u(f2)
        ud = fs.d_u(d2)
        ur = fs.r_u(r2)
        U = []
        U.append(uf)
        U.append(ud)
        U.append(ur)
        U = np.array(U)
        rmp2_tmp = fs.get_rmp(U)


        # if f2 == 0:
        #     rmp2_tmp = rmp2
        # else:
        #     uf = fs.survive_u(f2)
        #     ud = fs.d_u(d2)
        #     ur = fs.r_u(r2)
        #     U = []
        #     U.append(uf)
        #     U.append(ud)
        #     U.append(ur)
        #     U = np.array(U)
        #     rmp2_tmp = fs.get_rmp(U)





        rmp1 = rmp1 + f * (rmp1_tmp - rmp1)
        rmp2 = rmp2 + f * (rmp2_tmp - rmp2)

        # print(U,rmp2)
        # 删除后面部分,2NP个个体只保留前NP个个体
        for i in range(NP):
            population1[i]['child'] = False
            population1[i]['parent_cost'] = -1
            population2[i]['child'] = False
            population2[i]['parent_cost'] = -1
            population1.pop()
            population2.pop()

        # 更新最优
        for i in range(NP):
            if t1_gbest_F > population1[i]["cost"]:  # 更新任务1最优
                t1_gbest_F = population1[i]["cost"]
                t1_gbest_X = population1[i]["information"].copy()
            if t2_gbest_F > population2[i]["cost"]:  # 更新任务2最优
                t2_gbest_F = population2[i]["cost"]
                t2_gbest_X = population2[i]["information"].copy()
        # 记录当前代最优
        t1.append(t1_gbest_F)
        t2.append(t2_gbest_F)
        if g!=G-1:
            Rmp1.append(rmp1)
            Rmp2.append(rmp2)
        if g % 10 == 0:
            print(g, t1_gbest_F, t2_gbest_F)  # 输出当前代两个任务的最优值
    print(question, t1_gbest_F, t2_gbest_F)  # 输出question问题下两个任务的最优值
    return t1, t2, t1_gbest_F, t2_gbest_F,Rmp1, Rmp2
