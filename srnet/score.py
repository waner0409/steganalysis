import pandas as pd

# 读取CSV文件
df = pd.read_csv('./predictions.csv')

# 生成正确的标签（0和1交替）
correct_labels = [0 if i % 2 == 0 else 1 for i in range(len(df))]

# 计算准确率
accuracy = (df['Predicted_Label'] == correct_labels).mean()

print(f"预测的准确率是：{accuracy * 100}%")
