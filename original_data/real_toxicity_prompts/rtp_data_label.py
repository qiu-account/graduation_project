import pandas as pd
import json

# 加载数据
df = pd.read_json('D:/毕设/csr_project/original_data/real_toxicity_prompts.jsonl', lines=True)

# 查看数据的前几行，确认数据格式
print(df.head())

# 提取 toxicity 值
df['toxicity_prompt'] = df['prompt'].apply(
    lambda x: x.get('toxicity', 0) if isinstance(x, dict) else 0
)
df['toxicity_continuation'] = df['continuation'].apply(
    lambda x: x.get('toxicity', 0) if isinstance(x, dict) else 0
)

# 原有：综合 toxicity（保留）
df['toxicity'] = (df['toxicity_prompt'] + df['toxicity_continuation']) / 2

# ✅ 新增：Δ toxicity（模型行为指标）
df['delta_toxicity'] = df['toxicity_continuation'] - df['toxicity_prompt']

# 查看结果
print(df[['toxicity_prompt', 'toxicity_continuation', 'toxicity', 'delta_toxicity']].head())

# 原有：样本级 label（保留）
df['label'] = df['toxicity'].apply(lambda x: 1 if x > 0.5 else 0)

# 查看标注后的数据
print(df[['prompt', 'continuation', 'toxicity', 'delta_toxicity', 'label']].head())

# 保存数据
df.to_csv('real_toxicity_prompts_labeled1.csv', index=False)
