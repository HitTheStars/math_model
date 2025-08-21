import numpy as np
import matplotlib.pyplot as plt
import random

def generate_cities(n, seed=None):
    """生成n个随机城市坐标"""
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100

def distance_matrix(cities):
    """计算城市间距离矩阵"""
    n = len(cities)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = np.linalg.norm(cities[i] - cities[j])
    return D

def total_distance(route, D):
    """计算路线总距离"""
    dist = 0
    for i in range(len(route)):
        dist += D[route[i]][route[(i+1) % len(route)]]
    return dist

def two_opt_swap(route, i, j):
    """2-opt操作：反转route[i:j+1]"""
    new_route = route.copy()
    new_route[i:j+1] = reversed(new_route[i:j+1])
    return new_route

def generate_neighbor(route):
    """生成邻居路线（2-opt）"""
    i, j = sorted(random.sample(range(len(route)), 2))
    return two_opt_swap(route, i, j)

def simulated_annealing_tsp(cities, n_iterations, temp, alpha):
    """TSP的模拟退火求解"""
    n = len(cities)
    D = distance_matrix(cities)
    
    # 初始化：随机路线
    route = list(range(n))
    random.shuffle(route)
    best_route = route.copy()
    best_dist = total_distance(route, D)
    
    history = [best_dist]
    
    for iteration in range(n_iterations):
        # 生成邻居
        new_route = generate_neighbor(route)
        old_dist = total_distance(route, D)
        new_dist = total_distance(new_route, D)
        delta = new_dist - old_dist
        
        # Metropolis准则
        if delta < 0 or random.random() < np.exp(-delta / temp):
            route = new_route
            if new_dist < best_dist:
                best_route = new_route.copy()
                best_dist = new_dist
        
        # 降温
        temp *= alpha
        
        # 记录最佳距离
        if iteration % 10 == 0:
            history.append(best_dist)
    
    return best_route, best_dist, history

# ===== 运行示例 =====
cities = generate_cities(15)
best_route, best_dist, history = simulated_annealing_tsp(
    cities=cities,
    n_iterations=1000,
    temp=1000.0,
    alpha=0.99
)

print(f"最短路径长度: {best_dist:.2f}")
print(f"最优路线: {best_route}")

# ===== 可视化 =====
plt.figure(figsize=(15, 5))

# 子图1: 城市位置
plt.subplot(1, 3, 1)
plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=50)
for i, (x, y) in enumerate(cities):
    plt.annotate(i, (x, y), xytext=(5, 5), textcoords='offset points')
plt.title("城市位置")
plt.axis('equal')

# 子图2: 最优路径
plt.subplot(1, 3, 2)
ordered_cities = cities[best_route + [best_route[0]]]  # 闭合路径
plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'r-', linewidth=2)
plt.scatter(cities[:, 0], cities[:, 1], c='blue', s=50)
plt.title(f"最优路径 (长度: {best_dist:.2f})")
plt.axis('equal')

# 子图3: 收敛过程
plt.subplot(1, 3, 3)
plt.plot(history, 'b-')
plt.title("最佳距离收敛过程")
plt.xlabel("迭代次数")
plt.ylabel("路径长度")
plt.grid(True)

plt.tight_layout()
plt.show()