from datasets import load_dataset
import os

# 定义你要保存数据的文件夹路径
save_dir = "D:\毕设\csr_project\data\mmlu"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 加载 MMLU 数据集的指定子任务
mmlu_biology = load_dataset("cais/mmlu", "high_school_biology")  # 生命科学
mmlu_history = load_dataset("cais/mmlu", "high_school_us_history")  # 历史
mmlu_psychology = load_dataset("cais/mmlu", "high_school_psychology")  # 心理学

# 打印数据集的结构，检查可用的 splits
print(mmlu_biology)
print(mmlu_history)
print(mmlu_psychology)

# 使用 validation 数据集（如果没有 train 数据集）
mmlu_biology["validation"].to_pandas().to_csv(os.path.join(save_dir, "mmlu_biology_validation.csv"), index=False)
mmlu_history["validation"].to_pandas().to_csv(os.path.join(save_dir, "mmlu_history_validation.csv"), index=False)
mmlu_psychology["validation"].to_pandas().to_csv(os.path.join(save_dir, "mmlu_psychology_validation.csv"), index=False)

print("数据已成功保存！")
