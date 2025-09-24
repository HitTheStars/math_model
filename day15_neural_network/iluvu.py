import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 设置绘图风格
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# 1. 加载数据
iowa_file_path = r'G:\my_lovely_codes\math_model\day10_machine_learning\input\train.csv'
home_data = pd.read_csv(iowa_file_path)

# 2. 提取特征和标签
features_num = [
    'LotArea', 'MSSubClass', 'LotFrontage', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]

X = home_data[features_num].copy()
y = home_data['SalePrice'].copy()

# 3. 删除含缺失值的行（避免后续问题）

y = y.loc[X.index]

# 4. 数据预处理
preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
)

# 5. 划分训练/验证集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# 6. 缩放特征和标签
scale_factor = 100
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train_scaled = y_train / scale_factor
y_valid_scaled = y_valid / scale_factor

input_shape = [X_train.shape[1]]
print(f"✅ 输入形状: {input_shape}")

# 7. ✅ 修复模型结构：Dense → BatchNorm → Dropout
model = Sequential([
    layers.Dense(256, activation='relu', input_shape=input_shape),  # 👈 第一层：Dense
    layers.BatchNormalization(),                                  # 👈 第二层：BN
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),                         # 👈 不要写 input_shape
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(1)  # 输出层
])

# 8. 编译模型：使用 MSE 更稳定
model.compile(
    optimizer='adam',
    loss='mse',  # ✅ 改为 mse，训练更稳定
    metrics=['mae']  # ✅ 监控 mae，便于分析
)

# 9. 回调函数
early_stopping = callbacks.EarlyStopping(
    patience=25,
    min_delta=0.0005,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# 10. 训练模型
history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_valid, y_valid_scaled),
    batch_size=32,
    epochs=500,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 11. 绘制损失曲线
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title='Loss (MSE)')
plt.show()

history_df.loc[:, ['mae', 'val_mae']].plot(title='MAE (on scaled y)')
plt.show()

print(f"✅ 最小验证损失 (MSE): {history_df['val_loss'].min():.6f}")
print(f"✅ 最小验证 MAE (缩放后): {history_df['val_mae'].min():.6f}")

# 12. 预测并还原到真实价格
y_valid_pred_scaled = model.predict(X_valid).flatten()
y_valid_pred_real = y_valid_pred_scaled * scale_factor
mae_real = mean_absolute_error(y_valid, y_valid_pred_real)
print(f"✅ 最终验证集 MAE（真实价格）: ${mae_real:,.0f}")

# 13. 处理测试集（关键！必须用 transform，不是 fit_transform）
test_data_path = r'G:\my_lovely_codes\math_model\day10_machine_learning\input\test.csv'
test_data = pd.read_csv(test_data_path)

test_X = test_data[features_num].copy()

test_X_scaled = preprocessor.transform(test_X)  # ✅ 用训练好的 preprocessor！

test_preds_scaled = model.predict(test_X_scaled).flatten()
test_preds_real = test_preds_scaled * scale_factor

# 14. 生成提交文件

output = pd.DataFrame({
    'Id': test_data.loc[test_X.index, 'Id'],  # ✅ 用 clean 数据的索引取 Id
    'SalePrice': test_preds_real
})

output.to_csv('submission.csv', index=False)
print("✅ 提交文件已生成：submission.csv")