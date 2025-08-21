import numpy as np
import matplotlib.pyplot as plt
import random

def objective_function(x):
    """目标函数：f(x) = x * sin(x) + 0.5x，需最小化"""
    return x * np.sin(x) + 0.5 * x

def simulated_annealing(objective, bounds, n_iterations, step_size, temp, alpha):
    """
    模拟退火算法实现
    参数:
        objective: 目标函数 (需最小化)
        bounds: 搜索空间边界 [min, max]
        n_iterations: 总迭代次数
        step_size: 邻域扰动步长 (控制新解生成范围)
        temp: 初始温度
        alpha: 冷却速率 (0 < alpha < 1)
    返回:
        best: 最佳解
        best_eval: 最佳解的目标函数值
        history: 搜索过程记录 [(x, f(x)), ...]
    """
    # 步骤1: 初始化
    best = bounds[0] + random.uniform(0, 1) * (bounds[1] - bounds[0])  # 随机初始解
    best_eval = objective(best)  # 初始解的目标值
    current, current_eval = best, best_eval
    history = [(best, best_eval)]  # 记录搜索路径

    # 步骤2: 主循环 (迭代直到温度足够低)
    for i in range(n_iterations):
        # 生成新解: 在当前解附近添加随机扰动 (高斯噪声)
        candidate = current + random.gauss(0, step_size)
        # 确保新解在边界内
        candidate = max(min(candidate, bounds[1]), bounds[0])
        
        # 计算新解的目标值
        candidate_eval = objective(candidate)
        
        # 步骤3: Metropolis 准则
        # 情况1: 新解更优 (ΔE < 0) -> 一定接受
        if candidate_eval < current_eval:
            current, current_eval = candidate, candidate_eval
            # 如果比历史最佳更好，更新最佳解
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
                history.append((best, best_eval))
        # 情况2: 新解更差 (ΔE >= 0) -> 以概率接受
        else:
            delta = candidate_eval - current_eval  # ΔE
            # 计算接受概率 P = exp(-ΔE / T)
            accept_probability = np.exp(-delta / temp)
            # 以概率 accept_probability 接受较差解
            if random.random() < accept_probability:
                current, current_eval = candidate, candidate_eval
        
        # 步骤4: 降温 (每个迭代后降温)
        temp *= alpha  # 指数降温: T = T * α
        
        # 打印进度 (可选)
        if i % 100 == 0:
            print(f"Iteration {i}: T={temp:.3f}, f(x)={current_eval:.5f}, best={best_eval:.5f}")
    
    return best, best_eval, history

# ===== 参数设置 =====
bounds = [-10, 10]          # 搜索空间
n_iterations = 1000         # 总迭代次数
step_size = 0.5             # 邻域扰动步长 (控制探索范围)
initial_temp = 10.0         # 初始温度 (足够高)
alpha = 0.95                # 冷却速率 (0.8~0.99)

# ===== 运行模拟退火 =====
best_x, best_y, history = simulated_annealing(
    objective=objective_function,
    bounds=bounds,
    n_iterations=n_iterations,
    step_size=step_size,
    temp=initial_temp,
    alpha=alpha
)

print(f"\n全局最小值: x = {best_x:.5f}, f(x) = {best_y:.5f}")

# ===== 可视化结果 =====
# 1. 绘制目标函数曲线
x_vals = np.linspace(bounds[0], bounds[1], 500)
y_vals = objective_function(x_vals)

plt.figure(figsize=(12, 8))

# 子图1: 函数曲线 + 搜索路径
plt.subplot(2, 1, 1)
plt.plot(x_vals, y_vals, 'b-', label=r'$f(x) = x \cdot \sin(x) + 0.5x$')
plt.plot([p[0] for p in history], [p[1] for p in history], 'r.-', markersize=3, alpha=0.6, label='SA 路径')
plt.scatter(best_x, best_y, s=100, c='red', marker='*', label=f'全局最小值 ({best_x:.2f}, {best_y:.2f})')
plt.title('模拟退火优化过程')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# 子图2: 温度下降与最佳解收敛
temps = [initial_temp * (alpha ** i) for i in range(n_iterations)]
best_history = np.array([min([h[1] for h in history[:i+1]]) for i in range(len(history))])

plt.subplot(2, 1, 2)
plt.plot(temps, 'g-', label='Temperature (T)')
plt.plot(best_history, 'm-', label='Best f(x) so far')
plt.yscale('log')  # 温度用对数尺度更清晰
plt.title('温度下降与最佳解收敛')
plt.xlabel('Iteration')
plt.ylabel('Value (log scale)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()