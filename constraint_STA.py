import numpy as np
import random as rd


def f01(x):
    g = np.empty(2)
    f = 100 * np.square(x[1] - np.square(x[0])) + np.square(1 - x[0])
    g[0] = -x[0] - np.square(x[1])
    g[1] = -np.square(x[0]) - x[1]
    return f, g


def constraint_fitness(func, state):
    length = len(state)
    fit = np.empty([length, 2])
    for i in range(length):
        f, g = func(state[i])
        fit[i] = [f, np.sum(np.maximum(0, g))]
    return fit


def selection1(func, state):
    fit = constraint_fitness(func=func, state=state)
    f = fit[:, 1] + np.inf * np.square(fit[:, 2])
    index = np.argmin(f)
    return fit[index], state[index]


class Operators(object):
    def __init__(self, SE, alpha, beta, gamma, delta):
        self.SE = SE
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def rotate(self, best):
        n = best.size
        best = best.reshape(n, 1)  # 重新定义，则为局部变量
        a = np.tile(best, self.SE)
        b = np.dot(np.random.uniform(-1, 1, (self.SE * n, n)), best).reshape(n, self.SE)
        c = 1 / n / (np.linalg.norm(best) + 2e-16)  # 需要加上一个极小数
        y = a + self.alpha * c * b
        y = y.transpose()
        return y

    def axes(self, best):
        n = best.size
        A = np.zeros((n, self.SE))
        index = np.random.randint(0, n, (1, self.SE))
        A[index, list(range(self.SE))] = 1
        best = best.reshape(n, 1)
        a = np.tile(best, self.SE)
        b = np.array([rd.gauss(0, 1) for _ in range(n * self.SE)]).reshape(n, self.SE)
        c = self.delta * b * A * a
        y = a + c
        y = y.transpose()
        return y

    def expand(self, best):
        n = best.size  # 数组中元素的个数,不能用len，因为下一个best为矩阵形式，len[[1,2]] = 1
        best = best.reshape(n, 1)
        a = np.tile(best, self.SE)
        b = np.array([rd.gauss(0, 1) for _ in range(n * self.SE)]).reshape(n, self.SE)
        y = a + self.gamma * b * a
        y = y.transpose()
        return y

    def translate(self, old_best, new_best):
        n = old_best.size
        old_best = old_best.reshape(n, 1)
        new_best = new_best.reshape(n, 1)  # 定义局部变量
        diff = (new_best - old_best)
        a = np.tile(new_best, self.SE)
        b = self.beta / (np.linalg.norm(diff) + 2e-16)  # 需要加上一个极小值
        c = np.tile(np.random.uniform(0, 1, (1, self.SE)), n).reshape(n, self.SE) * np.tile(diff, self.SE)
        y = a + b * c
        y = y.transpose()
        return y


class Operation(Operators):
    def __init__(self, SE, alpha, beta, gamma, delta, selection, function, scope):
        super(Operation, self).__init__(SE, alpha, beta, gamma, delta)
        self.selection = selection
        self.function = function
        self.scope = scope
        self.dim = scope.shape[1]

    def constraint_operation(self, old_best, operator):
        state = np.empty([self.SE, self.dim])
        if operator == 'rotate':
            state = self.rotate(best=old_best)
        elif operator == 'expand':
            state = self.expand(best=old_best)
        elif operator == 'axesion':
            state = self.axes(best=old_best)
        new_best = self.selection(self.function, state)
        if new_best == old_best:
            return old_best
        else:
            state = self.translate(old_best=old_best, new_best=new_best)
            temp_best = self.selection(self.function, state)
            if temp_best == new_best:
                return new_best
            else:
                return temp_best



if __name__ == "__main__":
    op = Operators(SE=10, alpha=1, beta=1, gamma=1, delta=1)
    x = np.array([0.5, 0.25])
    y = op.expand(x)
    print(y)


