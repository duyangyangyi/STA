import numpy as np

# ################# 实例 ###########################
loc = [[0, 38.24, 20.42], [1, 39.57, 26.15], [2, 40.56, 25.32], [3, 36.26, 23.12],
       [4, 33.48, 10.54], [5, 37.56, 12.19], [6, 38.42, 13.11], [7, 37.52, 20.44],
       [8, 41.23, 9.10], [9, 41.17, 13.05], [10, 36.08, -5.21], [11, 38.47, 15.13],
       [12, 38.15, 15.35], [13, 37.51, 15.17], [14, 35.49, 14.32], [15, 39.36, 19.56]]
iteration = 400
SE = 20


# ################# 封装的方法 ###########################
def cal_distance(solution, location):
    t_solution = np.append(solution, solution[0])
    t_solution = t_solution.astype(np.int16)
    dist = 0
    for i in range(len(solution)):
        dist += np.linalg.norm(location[t_solution[i], 1:] - location[t_solution[i+1], 1:])
    return dist


def fitness_tsp(state, location):
    value = np.empty(state.shape[0])
    for i in range(state.shape[0]):
        value[i] = cal_distance(state[i], location)
    index = np.argmin(value)
    return state[index], value[index]


class Operators(object):
    def __init__(self, SE):
        self.SE = SE

    def shift(self, solution):
        dimension = solution.shape[0]

        state = np.empty([self.SE, dimension])
        for i in range(self.SE):
            temp_param = solution.copy()
            temp_list = np.arange(dimension)
            np.random.shuffle(temp_list)
            t = temp_list[:2]
            if t[0] < t[1]:
                temp = np.insert(temp_param, t[1]+1, temp_param[t[0]])  # 将t[0]数据插入t[1]数据的后面
                temp = np.delete(temp, t[0])  # 删除原t[0]数据
                temp_param = temp
            else:  # 同上
                temp = np.insert(temp_param, t[0]+1, temp_param[t[1]])
                temp = np.delete(temp, t[1])
                temp_param = temp
            state[i] = temp_param.copy()
        return state

    def swap(self, solution):
        dimension = solution.shape[0]

        state = np.empty([self.SE, dimension])
        for i in range(self.SE):
            temp_param = solution.copy()
            temp_list = np.arange(dimension)
            np.random.shuffle(temp_list)
            t = temp_list[:2]
            temp = temp_param[t[0]]
            temp_param[t[0]] = temp_param[t[1]]
            temp_param[t[1]] = temp  # 交换两个值
            state[i] = temp_param
        return state

    def symmetry(self, solution):
        dimension = solution.shape[0]

        state = np.empty([self.SE, dimension])
        for i in range(self.SE):
            temp_param = solution.copy()
            temp_list = np.arange(dimension)
            np.random.shuffle(temp_list)
            t = temp_list[:2]
            if t[0] < t[1]:
                temp = np.flipud(temp_param[t[0]:t[1]+1])  # 将切片部分进行翻转
                temp_param[t[0]:t[1]+1] = temp  # 将翻转的切片再赋值回原向量
            else:  # 同上
                temp = np.flipud(temp_param[t[1]:t[0]+1])
                temp_param[t[1]:t[0]+1] = temp
            state[i] = temp_param
        return state


class Operation(Operators):
    def __init__(self, SE, fitness, location):
        super(Operation, self).__init__(SE)
        self.fitness = fitness
        self.location = np.array(location)

    def operation(self, best, fbest, operator):
        if isinstance(best, list):
            best = np.array(best)
        dimension = best.shape[0]
        state = np.empty([self.SE, dimension])
        if operator == 'shift':
            state = self.shift(best)
        elif operator == 'swap':
            state = self.swap(best)
        elif operator == 'symmetry':
            state = self.symmetry(best)
        t_best, t_fbest = self.fitness(state, self.location)
        if t_fbest < fbest:
            return t_best, t_fbest
        else:
            return best, fbest


class D_STA(Operation):
    def __init__(self, SE, fitness, iterations, location):
        super(D_STA, self).__init__(SE, fitness, location)
        self.iterations = iterations
        self.history = np.empty(iterations)
        self.best = None
        self.fbest = None
        self.state = None
        self.init_self(SE=SE, fitness=fitness)

    def init_self(self, SE, fitness):
        dimension = self.location.shape[0]
        self.state = np.empty([SE, dimension])
        for i in range(SE):
            temp_list = np.arange(dimension)
            np.random.shuffle(temp_list)
            self.state[i] = temp_list
        self.best, self.fbest = fitness(self.state, self.location)

    def iteration(self):
        for i in range(self.iterations):
            self.best, self.fbest = self.operation(best=self.best, fbest=self.fbest, operator='swap')
            self.best, self.fbest = self.operation(best=self.best, fbest=self.fbest, operator='shift')
            self.best, self.fbest = self.operation(best=self.best, fbest=self.fbest, operator='symmetry')
            self.history[i] = self.fbest


# ################# 输出 ###########################
if __name__ == '__main__':

    sta = D_STA(SE=SE, fitness=fitness_tsp, iterations=iteration, location=loc)
    sta.iteration()
    print(sta.best)
    print(sta.fbest)

