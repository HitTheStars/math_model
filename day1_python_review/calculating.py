import opendatasets as od
import numpy as np
import csv

iris=od.download("https://www.kaggle.com/datasets/uciml/iris")

print("Iris dataset downloaded successfully.")
with open('iris/Iris.csv', 'r') as iris_file:
    lines = iris_file.readlines()  

print("Dataset location:", iris_file.name)

petal_lengths = []
petal_widths = []
for line in lines[1:]:            # lines[0]是标题行，跳过
    columns = line.strip().split(',')  # 去除换行符，按逗号分割
    petal_length = float(columns[3])   # 第4列转为浮点数
    petal_width = float(columns[4])
    petal_widths.append(petal_width)  # 添加到花瓣宽度列表  
    petal_lengths.append(petal_length)
print("花瓣长度列表:", petal_lengths)
print("前5个花瓣长度:", petal_lengths[:5])
print("花瓣长度的平均值:", np.mean(petal_lengths))
print("花瓣长度的标准差:", np.std(petal_lengths))
print("花瓣长度的最大值:", np.max(petal_lengths))
print("花瓣长度的最小值:", np.min(petal_lengths))       
print("花瓣长度的中位数:", np.median(petal_lengths))    
print("花瓣长度的方差:", np.var(petal_lengths))
print("花瓣长度的四分位数:", np.percentile(petal_lengths, [25, 50, 75]))  # 计算四分位数                                
print("花瓣宽度列表:", petal_widths)
print("花瓣宽度平均值：",np.mean(petal_widths))