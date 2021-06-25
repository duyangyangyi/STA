
'''**************************  实例   ***********************'''
from functools import reduce
import math
from operator import mul
import numpy as np


def Griewank(s):
    t1 = sum(map(lambda x: 1/4000 * x**2, s))
    n = len(s)
    t2 = map(lambda x, y: math.cos(x/np.sqrt(y)), s, range(1, n+1))
    t3 = reduce(mul, t2)
    return t1 - t3 + 1


SE = 30
Dim = 5
Range = np.tile([[-30], [30]], Dim)
Iterations = 100
Best0 = np.array(Range[0, :] + (Range[1, :]-Range[0, :]*np.random.uniform(0, 1, (1, Dim))))
'''************************************************************************************************'''


'''*******************  封装的方法   **************************'''
import numpy as np
import random as rd


def fitness(funfcn, State):
    SE = State.shape[0]
    fState = np.empty((SE, 1))  # 函数empty创建一个内容随机并且依赖与内存状态的数组
    fState = list(map(funfcn, State))  # 调用
    fGBest = np.min(fState)
    Best = State[fState.index(fGBest)]  # 这个列表中第一次此值的索引
    return Best, fGBest


def op_axes(Best, SE, delta):
    n = Best.size
    A = np.zeros((n, SE))
    index = np.random.randint(0, n, (1, SE))
    A[index, list(range(SE))] = 1
    Best = Best.reshape(n, 1)
    a = np.tile(Best, SE)
    b = np.array([rd.gauss(0, 1) for _ in range(n*SE)]).reshape(n, SE)
    c = delta*b*A*a
    y = a + c
    y = y.transpose()
    return y


def op_expand(Best, SE, gamma):
    n = Best.size # 数组中元素的个数,不能用len，因为下一个best为矩阵形式，len[[1,2]] = 1
    Best = Best.reshape(n, 1)
    a = np.tile(Best, SE)
    b = np.array([rd.gauss(0, 1) for _ in range(n*SE)]).reshape(n, SE)
    y = a + gamma * b * a
    y = y.transpose()
    return y


def op_rotate(Best, SE, alpha):
    n = Best.size
    Best = Best.reshape(n, 1) # 重新定义，则为局部变量
    a = np.tile(Best, SE)
    b = np.dot(np.random.uniform(-1, 1, (SE*n, n)), Best).reshape(n, SE)
    c = 1/n/(np.linalg.norm(Best) + 2e-16) # 需要加上一个极小数
    y = a + alpha * c * b
    y = y.transpose()
    return y


def op_translate(oldBest, newBest, SE, beta):
    n = oldBest.size
    oldBest = oldBest.reshape(n, 1)
    newBest = newBest.reshape(n, 1)    # 定义局部变量
    # newBest.shape=(n,1)             # 实际为全局变量操作，在执行函数之后会产生永久性改变
    diff = (newBest - oldBest)
    a = np.tile(newBest, SE)
    b = beta/(np.linalg.norm(diff) + 2e-16) # 需要加上一个极小值
    c = np.tile(np.random.uniform(0, 1, (1, SE)), n).reshape(n, SE) * np.tile(diff, SE)
    y = a + b * c
    y = y.transpose()
    return y


def axesion(funfcn, Best, fBest, SE, Range, beta, delta):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    oldBest = Best
    State = op_axes(Best, SE, delta)
    changeRows = State > Pop_Ub
    State[changeRows] = Pop_Ub[changeRows]
    changeRows = State < Pop_Lb
    State[changeRows] = Pop_Lb[changeRows]
    newBest,fGBest = fitness(funfcn, State)
    if fGBest < fBest:
        fBest,Best = fGBest, newBest
        State = op_translate(oldBest, Best, SE, beta)
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        newBest, fGBest = fitness(funfcn, State)
        if fGBest < fBest:
            fBest, Best = fGBest, newBest
    return Best, fBest


def rotate(funfcn, Best, fBest, SE, Range, alpha, beta):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    oldBest = Best
    State = op_rotate(Best, SE, alpha)
    changeRows = State > Pop_Ub
    State[changeRows] = Pop_Ub[changeRows]
    changeRows = State < Pop_Lb
    State[changeRows] = Pop_Lb[changeRows]
    newBest, fGBest = fitness(funfcn, State)
    if fGBest < fBest:
        fBest, Best = fGBest, newBest
        State = op_translate(oldBest, Best, SE, beta)
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        newBest,fGBest = fitness(funfcn, State)
        if fGBest < fBest:
            fBest, Best = fGBest, newBest
    return Best, fBest


def expand(funfcn, Best, fBest, SE, Range, beta, gamma):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    oldBest = Best
    State = op_expand(Best, SE, gamma)
    changeRows = State > Pop_Ub
    State[changeRows] = Pop_Ub[changeRows]
    changeRows = State < Pop_Lb
    State[changeRows] = Pop_Lb[changeRows]
    newBest, fGBest = fitness(funfcn, State)
    if fGBest < fBest:
        fBest, Best = fGBest, newBest
        State = op_translate(oldBest, Best, SE, beta)
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        newBest, fGBest = fitness(funfcn, State)
        if fGBest < fBest:
            fBest, Best = fGBest, newBest
    return Best, fBest


def update_alpha(funfcn, Best, fBest, SE, Range, Omega):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    m = len(Omega)
    alpha = 1
    Best0 = Best
    for i in range(m):
        State = op_rotate(Best0, SE, Omega[i])
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        tempBest, tempfBest = fitness(funfcn, State)
        if tempfBest < fBest:
            Best = tempBest
            fBest = tempfBest
            alpha = Omega[i]
    return Best, alpha


def update_delta(funfcn, Best, fBest, SE, Range, Omega):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    m = len(Omega)
    delta = 1
    Best0 = Best
    for i in range(m):
        State = op_axes(Best0, SE, Omega[i])
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        tempBest, tempfBest = fitness(funfcn, State)
        if tempfBest < fBest:
            Best = tempBest
            fBest = tempfBest
            delta = Omega[i]
    return Best, delta


def update_gamma(funfcn, Best, fBest, SE, Range, Omega):
    Pop_Lb = np.tile(Range[0], (SE, 1))
    Pop_Ub = np.tile(Range[1], (SE, 1))
    m = len(Omega)
    gamma = 1
    Best0 = Best
    for i in range(m):
        State = op_expand(Best0, SE, Omega[i])
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        tempBest, tempfBest = fitness(funfcn, State)
        if tempfBest < fBest:
            Best = tempBest
            fBest = tempfBest
            gamma = Omega[i]
    return Best, gamma


def rotate_w(funfcn, Best, fBest, SE, Range, Omega):
    Best, alpha = update_alpha(funfcn, Best, fBest, SE, Range, Omega)
    for i in range(10):
        Best, fBest = rotate(funfcn=funfcn, Best=Best, fBest=fBest, SE=SE, Range=Range, alpha=alpha, beta=1)
    return Best, fBest


def expand_w(funfcn, Best, fBest, SE, Range, Omega):
    Best, gamma = update_gamma(funfcn, Best, fBest, SE, Range, Omega)
    for i in range(10):
        Best, fBest = expand(funfcn=funfcn, Best=Best, fBest=fBest, SE=SE, Range=Range, gamma=gamma, beta=1)
    return Best, fBest


def axesion_w(funfcn, Best, fBest, SE, Range, Omega):
    Best, delta = update_delta(funfcn, Best, fBest, SE, Range, Omega)
    for i in range(10):
        Best, fBest = axesion(funfcn=funfcn, Best=Best, fBest=fBest, SE=SE, Range=Range, delta=delta, beta=1)
    return Best, fBest


def STA(funfcn, Best, SE, Range, Iterations):

    Omega = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    history = np.empty((1, Iterations))
    fBest = funfcn(Best[0])

    for iter in range(Iterations):
        Best, fBest = expand_w(funfcn, Best, fBest, SE, Range, Omega)
        Best, fBest = rotate_w(funfcn, Best, fBest, SE, Range, Omega)
        Best, fBest = axesion_w(funfcn, Best, fBest, SE, Range, Omega)
        history[0][iter] = fBest
    return Best, fBest, history
'''*********************************************************************'''
import matplotlib.pyplot as plt
'''********************************  输出   *****************************'''
if __name__ == '__main__':
    xmin, fxmin, history = STA(Griewank, Best0, SE, Range, Iterations)
    print(xmin, fxmin)



