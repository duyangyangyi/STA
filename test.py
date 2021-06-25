
def bubble_sort1(pareto_front, position):
    count = len(pareto_front)
    for i in range(0, count):
        for j in range(i + 1, count):
            if pareto_front[i].solution[position] > pareto_front[j].solution[position]:
                pareto_front[i], pareto_front[j] = pareto_front[j], pareto_front[i]


def bubble_sort2(pareto_front):
    count = len(pareto_front)
    for i in range(0, count):
        for j in range(i + 1, count):
            if pareto_front[i].crowding_distance > pareto_front[j].crowding_distance:
                pareto_front[i], pareto_front[j] = pareto_front[j], pareto_front[i]


class ParetoSolution(object):
    """
    自定义Pareto解，三个属性，Pareto排名，被支配解的的个数，支配解的集合，拥挤度
    """
    def __init__(self, solution):
        self.solution = solution  # 目标值向量
        self.pareto_rank = 0  # Pareto排名
        self.dominated_mun = 0  # 被支配个数
        self.crowding_distance = 0  # 拥挤度
        self.dominate_solution = []  # 支配解集
        self.sub_distance = []

    def clean(self):
        """
        在非支配排序完毕后调用，清空支配解集和被支配解的个数节省内存
        :return: None
        """
        self.dominate_solution = []
        self.sub_distance = []


def pareto_rank(pareto_solutions):
    """
    快速非支配排序
    :param pareto_solutions: 未排好序的Pareto解集
    :return: 排好序的Pareto解集
    """
    pareto_fronts = []  # Pareto前沿
    current_front = []  # 存放当前Pareto前沿解集
    front_num = 1  # Pareto第front_num前沿
    """ 遍历解集，找出Pareto第一前沿解集以及每一个解的被支配解的个数dominated_mun和支配解集dominate_solution """
    for pareto_solution in pareto_solutions:  # 遍历解集中每一个解
        for pareto_another_s in pareto_solutions:  # 与解集中的解进行比较
            dom_less, dom_equal, dom_more = 0, 0, 0
            for s_index in range(len(pareto_solution.solution)):  # 比较解中的每一个目标值
                if pareto_solution.solution[s_index] < pareto_another_s.solution[s_index]:
                    dom_less += 1
                elif pareto_solution.solution[s_index] == pareto_another_s.solution[s_index]:
                    dom_equal += 1
                else:
                    dom_more += 1
            if dom_less == 0 and dom_equal != len(pareto_solution.solution):  # 被解支配
                pareto_solution.dominated_mun += 1
            elif dom_more == 0 and dom_equal != len(pareto_solution.solution):  # 支配解
                pareto_solution.dominate_solution.append(pareto_another_s)
        if pareto_solution.dominated_mun == 0:  # Pareto第一前沿
            pareto_solution.pareto_rank = front_num
            current_front.append(pareto_solution)  # 加入当前Pareto前沿解集
    pareto_fronts.append(current_front)

    """ 处理剩下的解判断处于的Pareto前沿"""
    while len(current_front) != 0:  # 直到当前Pareto前沿解集为空集，退出循环
        front_num += 1
        temp_front = []
        for pareto_solution in current_front:  # 遍历当前Pareto前沿解集每一个解
            for dominate_solution in pareto_solution.dominate_solution:  # 遍历解中支配的解
                dominate_solution.dominated_mun -= 1  # 被支配解个数减一
                if dominate_solution.dominated_mun == 0:  # 新的Pareto前沿解
                    dominate_solution.pareto_rank = front_num
                    temp_front.append(dominate_solution)
        pareto_fronts.append(temp_front)
        current_front = temp_front  # 更新当前Pareto前沿解集

    """清空内存"""
    for pareto_solution in pareto_solutions:
        pareto_solution.clean()
    del pareto_fronts[-1]
    return pareto_fronts


def crowding_distance(pareto_fronts):
    """
    计算拥挤度距离
    """
    """遍历每一个Pareto前沿面"""
    for pareto_front in pareto_fronts:
        """遍历目标向量的每一个目标值"""
        for solution_mun in range(len(pareto_front[0].solution)):
            bubble_sort1(pareto_front, solution_mun)  # 按当前目标排序
            """遍历当前目标排序号的每一个解"""
            for index in range(len(pareto_front)):
                if index == 0 or index == (len(pareto_front)-1):  # 判断最大最小值
                    pareto_front[index].sub_distance.append(float("inf"))
                else:  # 计算每一个目标sub_distance
                    temp1 = pareto_front[index+1].solution[solution_mun] - pareto_front[index-1].solution[solution_mun]
                    temp2 = pareto_front[0].solution[solution_mun] - pareto_front[len(pareto_front)-1].solution[solution_mun]
                    sub_distance = temp1/temp2
                    pareto_front[index].sub_distance.append(sub_distance)
        """遍历每一个前沿中的解计算拥挤度"""
        for solution in pareto_front:
            solution.crowding_distance = sum(solution.sub_distance)  # 计算每一个解的crowding_distance
            solution.clean()
        bubble_sort2(pareto_front)
    return pareto_fronts


# s1 = ParetoSolution([3, 4, 5])
# s2 = ParetoSolution([4, 4, 5])
# s3 = ParetoSolution([5, 3, 5])
# s4 = ParetoSolution([5, 4, 5])
# solutions = [s1, s2, s3, s4]
# front = pareto_rank(solutions)
# print(len(front))


