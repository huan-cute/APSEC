import pandas as pd  # 用于数据读取与数据框(DataFrame)操作
from sklearn.metrics import precision_score, recall_score, f1_score  # 常用分类评估指标
from tqdm import tqdm  # 显示循环进度条
from openai import OpenAI  # DeepSeek R1 兼容的客户端类 (与官方 OpenAI SDK 接口一致)

# ==============================
# 配置与初始化
# ==============================

# ⚠️ 在实际使用时请务必替换为你自己的 DeepSeek API Key；
# 建议将密钥放在环境变量中，再通过 os.getenv("DEEPSEEK_API_KEY") 获取，避免明文写入代码。
client = OpenAI(
    api_key="your-api-key",
    base_url="your_url"

)

# ==============================
# Prompt 构造函数
# ==============================

def prompt_for_relation(requirements_text: str, code_text: str) -> str:
    """按照 DeepSeek 官方建议构造提示词 (Prompt)。

    参数:
        requirements_text (str): 需求或用例描述文本。
        code_text (str): 代码文本片段。

    返回:
        str: 构造好的 prompt 字符串。只能要求模型输出 "Yes" 或 "No"。
    """
    return (
        f"""Determine whether this requirement document is related to the code file. Answer only \"Yes\" or \"No\":\n\n"
        f"Requirements: {requirements_text}\n"
        f"Code: {code_text}\n\n"
        f"Answer:"""
    )

# ==============================
# 模型调用封装
# ==============================

def generate_response(prompt: str) -> str:
    """调用 DeepSeek R1 生成模型判断文本对是否相关。

    参数:
        prompt (str): 构造好的 prompt。

    返回:
        str: "yes" / "no" (统一为小写)。
    """
    try:
        # ChatCompletion 接口调用，保持与 OpenAI 官方 SDK 兼容
        response = client.chat.completions.create(
            # model="deepseek-chat",  # DeepSeek R1 模型名称
            # model = "meta-llama/llama-3.3-8b-instruct:free",
            # model = "google/gemini-2.5-flash-preview-05-20",
            # model= "deepseek/deepseek-r1-0528-qwen3-8b:free",
            model="google/gemini-2.5-pro-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a judge in the field of software traceability link recovery"
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=3,      # 只需要 "Yes" 或 "No"，3 个 token 足够
            temperature=0.0,   # 设为 0 保证确定性
            stop=["\n"]       # 遇到换行即停止，避免输出额外内容
        )

        # 从返回结果中提取模型输出内容并做简单归一化
        answer = response.choices[0].message.content.strip().lower()
        return "yes" if "yes" in answer else "no"

    except Exception as e:
        # 捕获并打印异常信息，生产环境可以记录日志
        print(f"API Error: {e}")
        # 出错时返回 "no"，相当于保守负例判断
        return "no"

# ==============================
# 上层逻辑封装
# ==============================

def is_related(issue_text: str, commit_text: str) -> int:
    """判定给定的需求文本和代码文本是否相关。返回 1 (相关) / 0 (不相关)。"""
    prompt = prompt_for_relation(issue_text, commit_text)
    response = generate_response(prompt)
    return 1 if response == "yes" else 0

# ==============================
# 指标计算
# ==============================

def calculate_metrics(csv_file: str):
    """读取数据集 CSV，随机抽样 10% 进行评估，计算 P/R/F1。

    CSV 文件必须包含以下列：
        - uc_text: 用例或需求描述
        - cc_text: 代码文本 (可为代码文件内容或片段)
        - label: 真值标签 (1 表示相关, 0 表示不相关)

    返回:
        tuple: (precision, recall, f1)
    """
    # 读取整个数据集
    df = pd.read_csv(csv_file)

    # 按 10% 比例随机采样，固定 random_state 以保证可复现
    eval_df = df.sample(frac=0.1, random_state=42)

    # 提取真值标签
    true_labels = eval_df['label'].tolist()
    predicted_labels = []

    # tqdm 用于展示进度条，iterrows() 逐行遍历评价样本
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing evaluation rows"):
        prediction = is_related(row['req'], row['code'])
        predicted_labels.append(prediction)

    # 计算 Precision / Recall / F1
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return precision, recall, f1

# ==============================
# 结果保存
# ==============================

def save_results_to_excel(results: list, output_file: str):
    """将评估结果保存到 Excel 文件 (每个数据集一行)。"""
    results_df = pd.DataFrame(results, columns=['dataset', 'precision', 'recall', 'f1'])
    # float_format 保留 4 位小数
    results_df.to_excel(output_file, index=False, float_format="%.4f")

# ==============================
# 主函数入口 (脚本执行时触发)
# ==============================

if __name__ == '__main__':
    # 待评估的项目列表，可按需修改或通过命令行参数传入
    datasets = ['Maven']

    for dataset_name in datasets:
        # 构造数据集路径 (假设目录结构: datasets/<项目名>/<项目名>_uc_cc.csv)
        csv_file = f"D:/jh_code/test/test/test1/csvfile/{dataset_name}/requirement_code_pairs_{dataset_name}.csv"
        # output_file = f"evaluation_results_{dataset_name}.xlsx"  # 输出文件名
        # input_csv = f"D:/jh_code/test/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}.csv"
        # output_csv = f"D:/jh_code/test/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}_with_reason.csv"

        print(f"Evaluating {csv_file} ...")

        # 计算评估指标
        precision, recall, f1 = calculate_metrics(csv_file)

        # 打印到控制台
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # # 写入 Excel
        # results = [(dataset_name, precision, recall, f1)]
        # save_results_to_excel(results, output_file)
        # print(f"Results saved to {output_file}\n")
