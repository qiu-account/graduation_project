# 完整的GCAV CAV训练代码 - 针对有害行为数据集
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os
import csv
from tqdm import tqdm

# ========== 配置 ==========
MODEL_PATH = "models\Qwen2-1.5B"
DATA_FILE = "JailbreakBench/harmful-behaviors.csv"  # 您的数据集
SAVE_DIR = "artifacts/toxicity_cav"
OUTPUT_FILE = "generated_outputs.csv"  # 保存生成内容
DEVICE = "cpu"
LAYER = 16  # 可以根据需要调整

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 加载模型 ==========
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 设置pad_token如果不存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32
).to(DEVICE)
model.eval()

# ========== 加载数据 ==========
print("加载数据...")
df = pd.read_csv(DATA_FILE)
print(f"数据: {len(df)}条有害提问")
print(f"列名: {df.columns.tolist()}")

# ========== 定义引导前缀 ==========
# 注意：这里我们需要为每个提问生成两种版本：有毒和无毒
# 由于原始数据都是有害提问，我们需要：
# 1. 用有毒前缀直接生成（有毒版本）
# 2. 用无毒前缀重新构造问题生成（无毒版本）

# 有毒引导前缀 - 模型倾向于生成有害内容
TOXIC_PREFIX = "Sure, here is"  # 原始数据中使用的格式

# 无毒引导前缀 - 模型倾向于拒绝或给出无害回应
NON_TOXIC_PREFIX = "I'm sorry, I cannot"

# ========== 提取激活并生成输出 ==========
X, y = [], []
generated_outputs = []

print("\n处理数据并生成激活...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    goal = row["Goal"]
    target = row["Target"]
    behavior = row["Behavior"]

    # 使用原始Goal作为提示（有毒版本）
    prompt = goal

    # ===== 1. 有毒版本 =====
    toxic_text = f"User: {prompt}\nAssistant: {TOXIC_PREFIX}"

    with torch.no_grad():
        # 获取激活
        inputs = tokenizer(toxic_text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER]

        # 取最后一个token的激活
        activation = hidden[0, -1].cpu().numpy()
        X.append(activation)
        y.append(1)  # 有毒标签

        # 生成完整回复（用于保存）
        if idx < 5:  # 只生成前5个作为示例，避免太多输出
            gen_outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            toxic_response = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        else:
            toxic_response = "生成跳过（只保存前5个示例）"

    # ===== 2. 无毒版本 =====
    # 重新构造问题，引导模型拒绝
    non_toxic_text = f"User: {prompt}\nAssistant: {NON_TOXIC_PREFIX}"

    with torch.no_grad():
        # 获取激活
        inputs = tokenizer(non_toxic_text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER]

        # 取最后一个token的激活
        activation = hidden[0, -1].cpu().numpy()
        X.append(activation)
        y.append(0)  # 无毒标签

        # 生成完整回复
        if idx < 5:
            gen_outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            non_toxic_response = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        else:
            non_toxic_response = "生成跳过（只保存前5个示例）"

    # 保存生成的内容
    generated_outputs.append({
        'index': idx,
        'goal': goal,
        'behavior': behavior,
        'category': row['Category'],
        'toxic_prefix': TOXIC_PREFIX,
        'non_toxic_prefix': NON_TOXIC_PREFIX,
        'toxic_generated': toxic_response,
        'non_toxic_generated': non_toxic_response,
        'toxic_label': 1,
        'non_toxic_label': 0
    })

# 转换为数组
X = np.array(X)
y = np.array(y)

print(f"\n激活形状: {X.shape}")
print(f"有毒样本: {sum(y == 1)}, 无毒样本: {sum(y == 0)}")

# ========== 保存生成的内容 ==========
print("\n保存生成的内容...")
output_df = pd.DataFrame(generated_outputs)
output_csv_path = os.path.join(SAVE_DIR, OUTPUT_FILE)
output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ 生成内容已保存: {output_csv_path}")

# 打印前几个示例
print("\n=== 前3个示例 ===")
for i in range(min(3, len(output_df))):
    print(f"\n示例 {i + 1}:")
    print(f"目标: {output_df.iloc[i]['goal']}")
    print(f"有毒生成（前100字符）: {output_df.iloc[i]['toxic_generated'][:100]}...")
    print(f"无毒生成（前100字符）: {output_df.iloc[i]['non_toxic_generated'][:100]}...")
    print("-" * 80)

# ========== 训练CAV ==========
print("\n训练CAV...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=42,
    solver='liblinear'
)
clf.fit(X_train, y_train)

# ========== 评估 ==========
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"\n=== 评估结果 ===")
print(f"交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"训练准确率: {train_acc:.4f}")
print(f"测试准确率: {test_acc:.4f}")

# ========== 提取CAV向量 ==========
w = clf.coef_[0]
b = clf.intercept_[0]
v = w / np.linalg.norm(w)  # 归一化

# 计算投影
projections = np.dot(X, v)
toxic_projs = projections[y == 1]
non_toxic_projs = projections[y == 0]

print(f"\n=== 投影分析 ===")
print(f"有毒投影均值: {toxic_projs.mean():.4f}")
print(f"无毒投影均值: {non_toxic_projs.mean():.4f}")
print(f"投影差异: {toxic_projs.mean() - non_toxic_projs.mean():.4f}")

# 计算统计显著性
from scipy import stats

t_stat, p_value = stats.ttest_ind(toxic_projs, non_toxic_projs)
print(f"T检验 p值: {p_value:.6f}")

# ========== 保存CAV ==========
metadata = {
    'layer': LAYER,
    'cav_vector': v,
    'intercept': b,
    'weights': w,
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'train_acc': float(train_acc),
    'test_acc': float(test_acc),
    'toxic_proj_mean': float(toxic_projs.mean()),
    'non_toxic_proj_mean': float(non_toxic_projs.mean()),
    'p_value': float(p_value),
    'toxic_prefix': TOXIC_PREFIX,
    'non_toxic_prefix': NON_TOXIC_PREFIX,
    'model_path': MODEL_PATH,
    'note': '毒性CAV - 使用引导前缀训练'
}

save_path = os.path.join(SAVE_DIR, 'toxicity_cav.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"\n✅ CAV已保存: {save_path}")

# ========== 可视化投影分布 ==========
import matplotlib.pyplot as plt

# ========== 可视化投影分布（修复中文显示问题） ==========
import matplotlib.pyplot as plt

# 方法1: 设置中文字体（如果系统有中文字体）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    use_chinese = True
except:
    use_chinese = False
    print("警告: 未找到中文字体，将使用英文标签")

plt.figure(figsize=(12, 7))

# 绘制直方图
plt.hist(toxic_projs, alpha=0.6, label='有毒样本' if use_chinese else 'Toxic',
         bins=30, density=True, color='red', edgecolor='black', linewidth=0.5)
plt.hist(non_toxic_projs, alpha=0.6, label='无毒样本' if use_chinese else 'Non-toxic',
         bins=30, density=True, color='blue', edgecolor='black', linewidth=0.5)

# 添加均值和阈值线
toxic_mean = toxic_projs.mean()
non_toxic_mean = non_toxic_projs.mean()

plt.axvline(toxic_mean, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.axvline(non_toxic_mean, color='blue', linestyle='--', alpha=0.7, linewidth=2)
plt.axvline(0, color='green', linestyle=':', alpha=0.5, linewidth=1.5,
            label='分类阈值 (0)' if use_chinese else 'Threshold (0)')

# 设置标签和标题
xlabel = 'CAV投影值' if use_chinese else 'CAV Projection Value'
ylabel = '密度' if use_chinese else 'Density'
title = '有毒 vs 无毒的CAV投影分布' if use_chinese else 'CAV Projection Distribution: Toxic vs Non-toxic'

plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
plt.title(title, fontsize=16, fontweight='bold')

# 添加图例
plt.legend(fontsize=12, loc='upper right')

# 添加统计信息标注
stats_text_en = f"""Statistics:
Toxic mean: {toxic_mean:.2f}
Non-toxic mean: {non_toxic_mean:.2f}
Difference: {toxic_mean - non_toxic_mean:.2f}
p-value: {p_value:.6f}"""

stats_text_cn = f"""统计信息:
有毒均值: {toxic_mean:.2f}
无毒均值: {non_toxic_mean:.2f}
差异: {toxic_mean - non_toxic_mean:.2f}
p值: {p_value:.6f}"""

stats_text = stats_text_cn if use_chinese else stats_text_en
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 美化网格
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# 保存图像
plot_path = os.path.join(SAVE_DIR, 'projection_distribution.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 投影分布图已保存: {plot_path}")
plt.show()

# 额外：创建更简单的英文版本（避免字体问题）
plt.figure(figsize=(10, 6))
plt.hist(toxic_projs, alpha=0.5, label='Toxic', bins=30, density=True, color='red')
plt.hist(non_toxic_projs, alpha=0.5, label='Non-toxic', bins=30, density=True, color='blue')
plt.xlabel('CAV Projection Value')
plt.ylabel('Density')
plt.title('CAV Projection Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

simple_plot_path = os.path.join(SAVE_DIR, 'projection_distribution_en.png')
plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ 英文版投影分布图已保存: {simple_plot_path}")
plt.close()  # 不显示这个图，只保存

# ========== 生成分析报告 ==========
report = f"""
=== 毒性CAV训练报告 ===
数据集: {DATA_FILE}
数据条数: {len(df)}
模型: {MODEL_PATH}
层数: {LAYER}

数据统计:
- 总激活数: {len(X)}
- 有毒样本: {sum(y == 1)}
- 无毒样本: {sum(y == 0)}

引导前缀:
- 有毒: "{TOXIC_PREFIX}"
- 无毒: "{NON_TOXIC_PREFIX}"

CAV性能:
- 交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})
- 训练准确率: {train_acc:.4f}
- 测试准确率: {test_acc:.4f}
- T检验 p值: {p_value:.6f}

投影分析:
- 有毒投影均值: {toxic_projs.mean():.4f}
- 无毒投影均值: {non_toxic_projs.mean():.4f}
- 差异: {toxic_projs.mean() - non_toxic_projs.mean():.4f}

生成内容示例已保存至: {output_csv_path}
"""

report_path = os.path.join(SAVE_DIR, 'training_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ 训练报告已保存: {report_path}")
print(report)