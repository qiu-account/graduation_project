import pandas as pd
import os

# 读取原始 JailbreakBench 数据
harmful = pd.read_csv('harmful-behaviors.csv')
benign = pd.read_csv('benign-behaviors.csv')

# 打概念标签
harmful['concept'] = 1   # harmful
benign['concept'] = 0    # benign

# 合并
df = pd.concat([harmful, benign], ignore_index=True)

# 只保留 CAV 需要的字段
df_cav = df[['Goal', 'concept', 'Category', 'Behavior']].copy()

# 重命名字段，语义更清晰
df_cav = df_cav.rename(columns={'Goal': 'prompt'})

# 创建目录
os.makedirs('jailbreak_cav', exist_ok=True)

# ✅ 保存“一二步处理完成后的干净数据”
df_cav.to_csv(
    'jailbreak_cav/jailbreak_cav_full.csv',
    index=False
)

print("已保存 CAV 训练用完整数据集：")
print(df_cav.head())
print("总样本数：", len(df_cav))
print("harmful 比例：", df_cav['concept'].mean())
