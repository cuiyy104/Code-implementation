import numpy as np
import matplotlib.pyplot as plt


class DifferentialEvolution:
    """
    差分进化算法(DE)实现

    参数:
    func: 目标函数
    n_dim: 变量维度
    size_pop: 种群大小
    max_iter: 最大迭代次数
    lb: 变量下界
    ub: 变量上界
    F: 缩放因子
    CR: 交叉概率
    """

    def __init__(self, func, n_dim, size_pop=50, max_iter=200,
                 lb=-100, ub=100, F=0.5, CR=0.3):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.F = F
        self.CR = CR

        # 初始化种群
        self.population = np.random.uniform(lb, ub, (size_pop, n_dim))
        self.fitness = np.array([func(ind) for ind in self.population])

        # 初始化最优解
        self.best_index = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_index].copy()
        self.best_fitness = self.fitness[self.best_index]

        # 收敛曲线
        self.convergence_curve = []

    def mutate(self, population):
        """变异操作: DE/rand/1策略"""
        mutated = np.zeros_like(population)

        for i in range(self.size_pop):
            # 选择三个不同的个体
            idxs = [idx for idx in range(self.size_pop) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            # 变异操作
            mutated[i] = a + self.F * (b - c)

        return mutated

    def crossover(self, population, mutated):
        """交叉操作: 二项式交叉"""
        trial = np.zeros_like(population)

        for i in range(self.size_pop):
            # 随机选择维度确保至少有一个维度发生交叉
            j_rand = np.random.randint(0, self.n_dim)

            for j in range(self.n_dim):
                if np.random.rand() < self.CR or j == j_rand:
                    trial[i, j] = mutated[i, j]
                else:
                    trial[i, j] = population[i, j]

        return trial

    def boundary_check(self, trial):
        """边界处理"""
        for i in range(self.size_pop):
            for j in range(self.n_dim):
                if trial[i, j] < self.lb:
                    trial[i, j] = self.lb
                elif trial[i, j] > self.ub:
                    trial[i, j] = self.ub
        return trial

    def optimize(self):
        """执行优化过程"""
        for iter in range(self.max_iter):
            # 变异操作
            mutated = self.mutate(self.population)

            # 交叉操作
            trial = self.crossover(self.population, mutated)

            # 边界处理
            trial = self.boundary_check(trial)

            # 计算适应度
            trial_fitness = np.array([self.func(ind) for ind in trial])

            # 选择操作
            for i in range(self.size_pop):
                if trial_fitness[i] < self.fitness[i]:
                    self.population[i] = trial[i]
                    self.fitness[i] = trial_fitness[i]

                    # 更新全局最优解
                    if trial_fitness[i] < self.best_fitness:
                        self.best_solution = trial[i].copy()
                        self.best_fitness = trial_fitness[i]

            # 记录收敛曲线
            self.convergence_curve.append(self.best_fitness)

            if iter % 50 == 0:
                print(f'迭代次数: {iter}, 最优适应度: {self.best_fitness:.6e}')

        return self.best_solution, self.best_fitness, self.convergence_curve


# 测试函数
def sphere_function(x):
    return np.sum(x ** 2)


if __name__ == "__main__":
    # 参数设置
    n_dim = 10
    size_pop = 50
    max_iter = 1000
    lb = -100
    ub = 100

    # 创建优化器实例
    de = DifferentialEvolution(sphere_function, n_dim, size_pop, max_iter, lb, ub)

    # 运行优化
    best_solution, best_fitness, convergence_curve = de.optimize()

    print(f'优化完成!')
    print(f'最优解: {best_solution}')
    print(f'最优适应度: {best_fitness:.10f}')

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(convergence_curve, 'b-', linewidth=2)
    plt.title('DE')
    plt.xlabel('epoch')
    plt.ylabel('fitness')
    plt.grid(True)
    plt.show()
