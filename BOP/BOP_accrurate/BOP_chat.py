import torch
import numpy as np
import pandas as pd
import pickle
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# Config
# =========================================================
MODEL_PATH = r"D:\毕设\project1\models\Qwen2-1.5B"
CAV_PATH = r"D:\毕设\project_end\artifacts\toxicity_cav\toxicity_cav.pkl"
MMLU_PATH = "mmlu_psychology_validation.csv"   # ← 改成你的路径
DEVICE = "cpu"

INTERVENTION_LAYER = 16
ALPHA_LIST = [-1.0, -2.0, -2.671, -3.0, -4.0, -10.0]  # 多个负值干预强度，向安全方向

# =========================================================
# Load model
# =========================================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32
).to(DEVICE).eval()

# =========================================================
# Load CAV
# =========================================================
print("Loading CAV...")
with open(CAV_PATH, "rb") as f:
    cav = pickle.load(f)

v = torch.tensor(cav['cav_vector'], dtype=torch.float32).to(DEVICE)

# =========================================================
# Hook（支持全局 ALPHA）
# =========================================================
class FixedInterventionHook:
    def __init__(self, alpha):
        self.alpha = alpha

    def hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] += self.alpha * v  # 负值 ALPHA → 向安全方向
        return out

    def register(self):
        layer = model.model.layers[INTERVENTION_LAYER]
        self.handle = layer.register_forward_hook(self.hook)

    def remove(self):
        self.handle.remove()

# =========================================================
# 读取 MMLU 数据
# =========================================================
df = pd.read_csv(MMLU_PATH)  # 默认逗号分隔
print(df.columns)
print(f"Number of questions: {len(df)}")

# =========================================================
# Helper functions
# =========================================================
def parse_choices(choice_str):
    """从字符串中提取所有选项"""
    return re.findall(r"'(.*?)'", choice_str)

def build_prompt(row):
    """动态生成 prompt，支持任意数量选项"""
    choices = parse_choices(row["choices"])
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # 可扩展更多

    if len(choices) < 4:
        print(f"[Warning] 题目选项少于 4 个: {row.get('question', '')}")
        print(f"Choices: {choices}")

    prompt_lines = [f"{option_labels[i]}. {choice}" if i < len(option_labels) else f"{i+1}. {choice}"
                    for i, choice in enumerate(choices)]
    question_text = row.get('question', '')
    full_prompt = question_text + "\n" + "\n".join(prompt_lines) + "\nAnswer:"
    return full_prompt

def gold_letter(ans_index):
    """数字答案 → A/B/C/D"""
    return ["A", "B", "C", "D"][int(ans_index)]

def get_prediction(prompt, intervene=False, alpha=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    if intervene:
        hook = FixedInterventionHook(alpha)
        hook.register()

    with torch.no_grad():
        outputs = model(**inputs)

    if intervene:
        hook.remove()

    logits = outputs.logits[0, -1]
    choices = ['A', 'B', 'C', 'D']
    choice_ids = [tokenizer(c).input_ids[-1] for c in choices]
    scores = [logits[i].item() for i in choice_ids]

    return choices[np.argmax(scores)]

# =========================================================
# 计算基线准确率
# =========================================================
print("Calculating baseline accuracy...")
correct_base = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="Baseline"):
    prompt = build_prompt(row)
    gold = gold_letter(row["answer"])
    pred_base = get_prediction(prompt, intervene=False)
    correct_base += (pred_base == gold)
acc_base = correct_base / len(df)
print(f"\nBaseline Accuracy: {acc_base:.4f}\n")

# =========================================================
# 批量测试不同 ALPHA 干预
# =========================================================
results = []

for ALPHA in ALPHA_LIST:
    print(f"\n=== Testing ALPHA = {ALPHA} ===")
    correct_int = 0
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Intervene ALPHA={ALPHA}"):
        prompt = build_prompt(row)
        gold = gold_letter(row["answer"])
        pred_int = get_prediction(prompt, intervene=True, alpha=ALPHA)
        correct_int += (pred_int == gold)
        records.append([prompt, pred_int, gold])

    acc_int = correct_int / len(df)
    print(f"ALPHA = {ALPHA}, Intervened Accuracy: {acc_int:.4f}")
    results.append({"ALPHA": ALPHA, "Acc_base": acc_base, "Acc_intervened": acc_int})

    # 保存每个 ALPHA 的详细预测结果
    pd.DataFrame(records, columns=["prompt", "pred_intervened", "gold"])\
      .to_csv(f"BOP_outputs_psychology_ALPHA_{ALPHA}.csv", index=False)

# =========================================================
# 保存汇总结果
# =========================================================
pd.DataFrame(results).to_csv("BOP_summary_results.csv", index=False)
print("\nAll done! Summary saved to BOP_summary_results.csv")
