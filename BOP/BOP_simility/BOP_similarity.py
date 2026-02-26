import torch
import numpy as np
import pandas as pd
import pickle
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Config
# =========================================================
MODEL_PATH = r"D:\毕设\project_end\models\Qwen2-1.5B"
CAV_PATH = r"D:\毕设\project_end\artifacts\toxicity_cav\toxicity_cav.pkl"
MMLU_PATH = "mmlu_psychology_validation.csv"   # ← 改成你的路径
DEVICE = "cpu"

INTERVENTION_LAYER = 16
#ALPHA_LIST = [-2.671]  # 向安全方向
ALPHA_LIST = [-1.0, -2.0, -2.671, -3.0, -4.0, -10]

# =========================================================
# Load model
# =========================================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    output_hidden_states=True   # ← 关键，返回 hidden states
).to(DEVICE).eval()

# =========================================================
# Load CAV
# =========================================================
print("Loading CAV...")
with open(CAV_PATH, "rb") as f:
    cav = pickle.load(f)

v = torch.tensor(cav['cav_vector'], dtype=torch.float32).to(DEVICE)

# =========================================================
# Hook
# =========================================================
class FixedInterventionHook:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_hidden = None  # 保存干预后的 hidden state

    def hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] += self.alpha * v
        self.last_hidden = h[:, -1, :].detach().cpu()
        return out

    def register(self):
        layer = model.model.layers[INTERVENTION_LAYER]
        self.handle = layer.register_forward_hook(self.hook)

    def remove(self):
        self.handle.remove()

# =========================================================
# 读取 MMLU 数据
# =========================================================
df = pd.read_csv(MMLU_PATH)
print(df.columns)
print(f"Number of questions: {len(df)}")

def parse_choices(choice_str):
    return re.findall(r"'(.*?)'", choice_str)

def build_prompt(row):
    choices = parse_choices(row["choices"])
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    if len(choices) < 4:
        print(f"[Warning] 题目选项少于 4 个: {row.get('question', '')}")
        print(f"Choices: {choices}")
    prompt_lines = [f"{option_labels[i]}. {choice}" if i < len(option_labels) else f"{i+1}. {choice}"
                    for i, choice in enumerate(choices)]
    question_text = row.get('question', '')
    full_prompt = question_text + "\n" + "\n".join(prompt_lines) + "\nAnswer:"
    return full_prompt

def gold_letter(ans_index):
    return ["A", "B", "C", "D"][int(ans_index)]

# =========================================================
# 获取预测和 hidden state embedding
# =========================================================
def get_embedding(prompt, intervene=False, alpha=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    hook = None
    if intervene:
        hook = FixedInterventionHook(alpha)
        hook.register()
    with torch.no_grad():
        outputs = model(**inputs)
    if intervene:
        hook.remove()
        embedding = hook.last_hidden.squeeze(0).numpy()
    else:
        embedding = outputs.hidden_states[INTERVENTION_LAYER][:, -1, :].squeeze(0).cpu().numpy()
    return embedding

# =========================================================
# 批量计算 BOP embedding 相似度
# =========================================================
results = []

for ALPHA in ALPHA_LIST:
    print(f"\n=== Testing ALPHA = {ALPHA} ===")
    sim_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Intervene ALPHA={ALPHA}"):
        prompt = build_prompt(row)
        emb_base = get_embedding(prompt, intervene=False)
        emb_int = get_embedding(prompt, intervene=True, alpha=ALPHA)

        # 计算余弦相似度
        sim = cosine_similarity(emb_base.reshape(1, -1), emb_int.reshape(1, -1))[0, 0]
        sim_list.append(sim)

    mean_sim = np.mean(sim_list)
    print(f"ALPHA = {ALPHA}, Mean embedding similarity (BOP) = {mean_sim:.4f}")
    results.append({"ALPHA": ALPHA, "Mean_BOP_similarity": mean_sim})

    # 保存每个 ALPHA 的题目相似度
    pd.DataFrame({
        "prompt": df['question'],
        "embedding_similarity": sim_list
    }).to_csv(f"BOP_embedding_ALPHA_{ALPHA}.csv", index=False)

# 保存汇总结果
pd.DataFrame(results).to_csv("BOP_embedding_summary.csv", index=False)
print("\nAll done! Summary saved to BOP_embedding_summary.csv")

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 方法 1：从之前保存的汇总 CSV 读取
# -----------------------------
summary_csv = "BOP_embedding_summary.csv"  # 汇总文件
df_summary = pd.read_csv(summary_csv)

plt.figure(figsize=(6,4))
plt.plot(df_summary['ALPHA'], df_summary['Mean_BOP_similarity'], marker='o', linestyle='-')
plt.xlabel("ALPHA (干预强度)")
plt.ylabel("Mean BOP Embedding Similarity")
plt.title("干预强度 vs BOP 相似度")
plt.grid(True)
plt.ylim(0,1.05)
plt.xticks(df_summary['ALPHA'])
plt.show()

# -----------------------------
# 方法 2：如果你有列表字典数据
# -----------------------------
# results = [
#     {"ALPHA": -1.0, "Mean_BOP_similarity": 0.90},
#     {"ALPHA": -2.0, "Mean_BOP_similarity": 0.875},
#     {"ALPHA": -2.671, "Mean_BOP_similarity": 0.861},
#     {"ALPHA": -3.0, "Mean_BOP_similarity": 0.84},
#     {"ALPHA": -4.0, "Mean_BOP_similarity": 0.80},
# ]
# df_summary = pd.DataFrame(results)
# （画图同上）

