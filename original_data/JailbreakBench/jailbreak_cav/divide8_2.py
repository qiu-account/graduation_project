#train:test=8:2
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 读取 full 数据
df = pd.read_csv(
    r"jailbreak_cav_full.csv"
)

# 2. 分层抽样（保证 harmful / benign 比例一致）
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["concept"],
    random_state=42
)

# 3. 创建目录（如果不存在）
#os.makedirs("data/jailbreak_cav", exist_ok=True)

# 4. 保存
train_df.to_csv(
    "jailbreak_cav_train.csv",
    index=False
)
test_df.to_csv(
    "jailbreak_cav_test.csv",
    index=False
)

print("✅ CAV train / test 数据已生成")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
