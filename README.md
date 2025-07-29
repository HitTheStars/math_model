# 数学建模学习项目

这个仓库包含了数学建模相关的代码、数据集和学习资料，主要用于学习和实践数学建模技术，为全国大学生数学建模竞赛做准备。

## 项目结构

```
├── # environment.yml     # Conda环境配置文件
├── .vscode/             # VS Code配置文件夹
├── day1_python_review/  # Python基础复习代码
│   ├── calculating.py   # 数据处理示例（Iris数据集分析）
│   └── doing_basic_math.py # 基础数学计算示例
├── iris/                # Iris数据集文件夹
│   ├── Iris.csv         # Iris数据集CSV文件
│   └── database.sqlite  # Iris数据集SQLite数据库
├── kaggle.json         # Kaggle API配置文件
├── 学习掌握的目标.md    # 数学建模学习目标和技能清单
└── 学习日程表.md        # 30天学习计划和日程安排
```

## 环境配置

本项目使用Conda管理环境，主要依赖包括：

- Python 3.9
- NumPy
- Pandas
- SciPy
- Matplotlib
- PyTorch
- scikit-learn

可以使用以下命令创建环境：

```bash
conda env create -f "# environment.yml"
```

## 代码示例

### Iris数据集分析 (calculating.py)

这个脚本展示了如何：

1. 使用opendatasets从Kaggle下载Iris数据集
2. 读取CSV文件并提取花瓣长度和宽度数据
3. 计算基本统计量（平均值、标准差、最大值、最小值等）

示例用法：

```bash
python day1_python_review/calculating.py
```

### 基础数学计算 (doing_basic_math.py)

这个脚本展示了如何使用SciPy进行数值积分计算，计算了sin(x)从0到π的积分。

示例用法：

```bash
python day1_python_review/doing_basic_math.py
```

## 学习目标

本项目的学习目标包括以下几个方面：

1. **Python编程与核心科学计算栈**：Python基础语法、NumPy、Pandas、Matplotlib/Seaborn、SciPy
2. **数学建模核心算法与实现**：优化模型、统计与数据分析模型、图论与网络优化、微分方程/差分方程模型、评价与决策模型
3. **论文写作与辅助工具**：LaTeX、Markdown、图表优化、版本控制
4. **环境搭建与效率工具**：Anaconda/Miniconda、VSCode/PyCharm、Jupyter Notebook

详细的学习目标请参考 [学习掌握的目标.md](./学习掌握的目标.md)。

## 学习计划

本项目采用30天极限冲刺计划，分为以下几个阶段：

1. **第1-3天**：Python + 核心库「肌肉记忆」唤醒
2. **第4-7天**：可视化 + SciPy核心武器
3. **第8-21天**：核心算法突破 + 疯狂模拟赛
4. **第22-28天**：论文写作 + LaTeX + Git 极简生存
5. **第29-30天**：查漏补缺 + 心态调整

详细的学习计划请参考 [学习日程表.md](./学习日程表.md)。

## 数据集

### Iris数据集

Iris数据集是一个经典的机器学习数据集，包含了三种鸢尾花的测量数据：

- 特征：萼片长度、萼片宽度、花瓣长度、花瓣宽度（厘米）
- 类别：Iris-setosa、Iris-versicolor、Iris-virginica
- 样本数：150（每类50个样本）

## 使用Kaggle API

本项目使用Kaggle API下载数据集。使用前需要：

1. 在Kaggle网站上创建API令牌
2. 将kaggle.json文件放在项目根目录下

## 贡献

欢迎提交问题和改进建议！

## 许可

本项目采用MIT许可证。