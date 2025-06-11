import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  # 引入进度条库

# 读取 CSV
csv_path = "./ESC-50-master/meta/esc50.csv"
df = pd.read_csv(csv_path,index_col=3)

# 读取 npy 文件
data_dir = "./new_Feature"
X, y = [], []

# 读取数据并加入进度条
for file in tqdm(os.listdir(data_dir), desc="加载数据", unit="文件"):
    if file.endswith(".npy"):
        file_path = os.path.join(data_dir, file)
        filename_key = file.replace(".npy", ".wav")  # 添加 .wav 后缀以匹配 CSV

        # 检查 filename_key 是否在 CSV 中
        if filename_key in df["filename"].values:
            X.append(np.load(file_path).flatten())  # 展平数据
            y.append(df.loc[df["filename"] == filename_key, "target"].values[0])

# 转换为 numpy 数组
X = np.array(X)
y = np.array(y)

# **添加数据检查**
if len(X) == 0:
    raise ValueError("没有加载任何数据，请检查 CSV 和 npy 文件名是否匹配！")

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **加入 PCA**
pca = PCA(n_components=0.95)  # 选 95% 的信息
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 训练 SVM
svm_model = SVC(kernel="linear", C=1.0)

# 使用 tqdm 显示训练的进度
tqdm.write("开始训练模型...")
svm_model.fit(X_train, y_train)  # 训练模型

# 预测并显示进度
tqdm.write("开始预测...")
y_pred = []
for i in tqdm(range(len(X_test)), desc="预测进度", unit="样本"):
    y_pred.append(svm_model.predict([X_test[i]]))  # 逐个预测
y_pred = np.array(y_pred).flatten()  # 转换为 numpy 数组

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM + PCA 分类准确率: {accuracy:.2f}")

# 生成分类报告
report = classification_report(y_test, y_pred, output_dict=True)

# 将分类报告转换为 DataFrame
report_df = pd.DataFrame(report).transpose()

# 将分类报告保存为 CSV
report_csv_path = "classification_PCA_report.csv"
report_df.to_csv(report_csv_path, index=True)

print(f"分类报告已保存为 CSV 文件：{report_csv_path}")
