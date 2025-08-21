import numpy as np
from scipy.optimize import minimize, LinearConstraint

# 定义目标函数
def objective(x):
    return -((x[0] )**2 + (x[1])**2+ 3*(x[2])**2+4*(x[3])**2+2*(x[4])**2-8*x[0]-2*x[1]-3*x[2]-x[3]-2*x[4])


# 定义线性不等式约束
A = [[1, 1,1,1,1],
        [1,2,2,1,6],
        [2,1,6,0,0],
        [0,0,1,1,5],
        ]  # 线性不等式约束矩阵
B = [400,800,200,200]     # 线性不等式约束右侧值



# 定义变量边界
bounds = [(0, 99), (0, 99), (0, 99), (0, 99), (0, 99)]  # 0<= xi <=99

# 初始猜测值
x0 = np.zeros(5)
x0[0] = 50
x0[1] = 50
x0[2] = 50
x0[3] = 50
x0[4] = 50



# 定义约束条件
linear_constraint = LinearConstraint(A, -np.inf, B)  # 线性不等式约束

# 使用SLSQP算法求解
result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=[linear_constraint])



# 输出结果
print("最优解:", result.x)
print("目标函数的最大值:", -result.fun)