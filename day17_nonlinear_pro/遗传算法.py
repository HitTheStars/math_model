import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

from matplotlib.font_manager import FontProperties
chinese_font = FontProperties(fname='G:\\my_lovely_codes\\math_model\\ZH_ZN\\SourceHanSansSC-Regular.otf')
plt.rcParams['font.sans-serif'] = [chinese_font.get_name()]
plt.rcParams['axes.unicode_minus'] = False


def objective_function(x):
    """目标函数：f(x) = x * sin(x) + 0.5x，需最小化"""
    return x * np.sin(x) + 0.5 * x

def fitness_function(x):
    """适应度函数：目标函数的倒数（因为我们要最小化目标）"""
    # 添加1防止除以0，取负因为我们要最大化适应度
    return 1.0 / (1.0 + objective_function(x))

class GeneticAlgorithm:
    def __init__(self, bounds, pop_size=50, crossover_rate=0.85, 
                 mutation_rate=0.02, tournament_size=3, elitism_size=2):
        """
        遗传算法初始化
        参数:
            bounds: 搜索空间边界 [min, max]
            pop_size: 种群大小
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            tournament_size: 锦标赛选择的大小
            elitism_size: 精英保留数量
        """
        self.bounds = bounds
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        
        # 初始化种群（实数编码）
        self.population = [random.uniform(bounds[0], bounds[1]) for _ in range(pop_size)]
        self.fitness = [fitness_function(x) for x in self.population]
        
        # 记录最佳解
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.population[best_idx]
        self.best_fitness = self.fitness[best_idx]
        self.best_objective = objective_function(self.best_solution)
        
        # 用于记录进化过程
        self.history = [(self.best_solution, self.best_objective)]
    
    def tournament_selection(self):
        """锦标赛选择：随机选k个个体，取适应度最高的"""
        candidates = random.sample(range(self.pop_size), self.tournament_size)
        fitnesses = [self.fitness[i] for i in candidates]
        winner_idx = candidates[np.argmax(fitnesses)]
        return self.population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """算术交叉：产生两个加权平均的后代"""
        if random.random() < self.crossover_rate:
            # 算术交叉：z1 = α·x + (1-α)·y, z2 = (1-α)·x + α·y
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            
            # 确保在边界内
            child1 = max(min(child1, self.bounds[1]), self.bounds[0])
            child2 = max(min(child2, self.bounds[1]), self.bounds[0])
            return child1, child2
        else:
            return parent1, parent2
    
    def mutate(self, individual):
        """高斯变异：添加小的随机扰动"""
        if random.random() < self.mutation_rate:
            # 高斯扰动，标准差随迭代减小（模拟退火思想）
            sigma = (self.bounds[1] - self.bounds[0]) * 0.1
            mutant = individual + random.gauss(0, sigma)
            # 确保在边界内
            return max(min(mutant, self.bounds[1]), self.bounds[0])
        return individual
    
    def evolve(self, generations=100):
        """执行遗传算法进化"""
        for gen in range(generations):
            # 1. 精英保留：复制最佳个体
            new_population = []
            elite_indices = np.argsort(self.fitness)[-self.elitism_size:]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # 2. 选择、交叉、变异生成新种群
            while len(new_population) < self.pop_size:
                # 选择父母
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # 添加到新种群
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # 3. 评估新种群
            self.population = new_population[:self.pop_size]  # 确保种群大小
            self.fitness = [fitness_function(x) for x in self.population]
            
            # 4. 更新最佳解
            best_idx = np.argmax(self.fitness)
            current_best = self.population[best_idx]
            current_fitness = self.fitness[best_idx]
            current_objective = objective_function(current_best)
            
            if current_fitness > self.best_fitness:
                self.best_solution = current_best
                self.best_fitness = current_fitness
                self.best_objective = current_objective
            
            # 记录进化过程
            self.history.append((self.best_solution, self.best_objective))
            
            # 打印进度
            if gen % 10 == 0:
                print(f"Generation {gen}: Best x={current_best:.5f}, f(x)={current_objective:.5f}")
        
        return self.best_solution, self.best_objective, self.history

# ===== 参数设置 =====
bounds = [-10, 10]          # 搜索空间
pop_size = 50               # 种群大小
crossover_rate = 0.85       # 交叉概率
mutation_rate = 0.02        # 变异概率
tournament_size = 3         # 锦标赛大小
elitism_size = 2            # 精英保留数量
generations = 100           # 迭代次数

# ===== 运行遗传算法 =====
ga = GeneticAlgorithm(
    bounds=bounds,
    pop_size=pop_size,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    tournament_size=tournament_size,
    elitism_size=elitism_size
)

best_x, best_y, history = ga.evolve(generations=generations)

print(f"\n全局最小值: x = {best_x:.5f}, f(x) = {best_y:.5f}")

# ===== 可视化结果 =====
# 1. 绘制目标函数曲线
x_vals = np.linspace(bounds[0], bounds[1], 500)
y_vals = objective_function(x_vals)

plt.figure(figsize=(14, 10))

# 子图1: 函数曲线 + 种群分布（最后一代）
plt.subplot(2, 2, 1)
plt.plot(x_vals, y_vals, 'b-', label=r'$f(x) = x \cdot \sin(x) + 0.5x$')
plt.scatter(ga.population, [objective_function(x) for x in ga.population], 
            c='red', s=30, alpha=0.6, label='最终种群')
plt.scatter(best_x, best_y, s=100, c='green', marker='*', 
            label=f'全局最小值 ({best_x:.2f}, {best_y:.2f})')
plt.title('最终种群分布', fontproperties=chinese_font)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(prop=chinese_font)
plt.grid(True)
plt.xticks()
plt.yticks()


# 子图2: 进化过程（最佳解收敛）
plt.subplot(2, 2, 2)
gen_nums = range(len(history))
best_xs = [h[0] for h in history]
best_ys = [h[1] for h in history]

plt.plot(gen_nums, best_ys, 'm-', linewidth=2, label='最佳目标值')
plt.axhline(y=-8.65732, color='r', linestyle='--', label='理论全局最小值')
plt.title('最佳解收敛过程', fontproperties=chinese_font)
plt.xlabel('代数')
plt.ylabel('f(x)')
plt.legend(prop=chinese_font)
plt.grid(True)

# 子图3: 种群多样性变化
plt.subplot(2, 2, 3)
diversity = []
for gen in range(0, len(ga.history), max(1, len(ga.history)//20)):
    pop = [x for x, _ in history[:gen+1]]
    diversity.append(np.std(pop))

plt.plot(range(0, len(ga.history), max(1, len(ga.history)//20)), diversity, 'g-o')
plt.title('种群多样性变化', fontproperties=chinese_font)
plt.xlabel('代数')
plt.ylabel('标准差')
plt.grid(True)
plt.xticks()
plt.yticks()

# 子图4: 每代平均适应度
plt.subplot(2, 2, 4)
avg_fitness = []
for gen in range(generations):
    start_idx = gen * ga.pop_size
    end_idx = min((gen+1) * ga.pop_size, len(history))
    if start_idx < end_idx:
        gen_fitness = [objective_function(x) for x, _ in history[start_idx:end_idx]]
        avg_fitness.append(np.mean(gen_fitness))

plt.plot(range(len(avg_fitness)), avg_fitness, 'b-')
plt.title('每代平均目标值', fontproperties=chinese_font)
plt.xlabel('代数')
plt.ylabel('f(x)')
plt.grid(True)
plt.xticks()
plt.yticks()

plt.tight_layout()
plt.show()