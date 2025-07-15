# -*- coding: utf-8 -*-
# @Time : 2025/6/12 9:40
# @Author : lxf
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import time

# 配置Gemini API
client = OpenAI(
    api_key="your-api-key",
    base_url="your_url"
)
dataset = "itrust"
# 文件路径配置
input_csv = f"D:/jh_code/test/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}.csv"
output_csv = f"D:/jh_code/test/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}_with_gemini_reason.csv"


def generate_content_based_reason(req_text: str, code_text: str, label: int) -> str:
    """基于实际内容生成关联原因解释"""
    try:
        prompt = (
            f"Analyze this requirement-code pair and explain why they are {'related' if label else 'unrelated'} "
            f"based on their actual content. Focus on technical implementation details.\n\n"
            f"=== REQUIREMENT ===\n{req_text[:2000]}\n\n"  # 限制长度避免token超标
            f"=== CODE ===\n{code_text[:2000]}\n\n"
            "Provide a concise technical explanation (under 60 words):"
        )

        response = client.chat.completions.create(
            model="google/gemini-2.5-pro-preview",
            messages=[
                {"role": "system", "content": "You are a senior software engineer analyzing traceability links"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=1.0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API Error: {str(e)}")
        return "[Explanation generation failed]"


def process_csv_with_content():
    """处理CSV文件（基于内容生成原因后删除内容列）"""
    # 读取原始CSV
    df = pd.read_csv(input_csv)

    # 添加Reason列
    tqdm.pandas(desc="Generating reasons")
    df['Reason'] = df.progress_apply(
        lambda row: generate_content_based_reason(row['req'], row['code'], row['label']),
        axis=1
    )

    # 删除内容列
    new_df = df.drop(columns=['req', 'code'])

    # 保存新CSV
    new_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n处理完成！新文件已保存至: {output_csv}")
    print("最终列结构:", new_df.columns.tolist())


if __name__ == '__main__':
    process_csv_with_content()