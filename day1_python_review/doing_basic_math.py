# 直接复制粘贴！不要手敲！
from scipy import integrate
import numpy as np

# 求 sin(x) 从0到π的积分（答案=2）
result, err = integrate.quad(np.sin, 0, np.pi)  # 重点：np.sin后面没括号！
print("积分结果:", result, "应该接近2")