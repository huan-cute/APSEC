import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from tqdm import tqdm

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_texts(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,                      # 输入文本
            add_special_tokens=True,   # 添加特殊标记
            max_length=max_length,     # 截断的最大长度
            truncation=True,           # 显式截断
            pad_to_max_length=True,    # 填充到最大长度
            return_attention_mask=True,# 返回注意力掩码
            return_tensors='pt',       # 返回 PyTorch 张量
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(42)  # 设置随机种子以确保可重复性

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_length = 512  # RoBERTa的最大输入长度
    batch_size = 8  # 减小批次大小以减少内存使用
    dataset = "maven"
    # 读取CSV文件
    input_file = f'D:/jh_code/test/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}_with_deepseekR1_reason.csv'
    df = pd.read_csv(input_file)

    # 检查是否包含'Reason'列
    if 'Reason' not in df.columns:
        print("CSV文件中未找到'Reason'列")
        exit()

    # 提取'Reason'列
    reasons = df['Reason'].fillna('').tolist()  # 处理缺失值

    # 初始化 RoBERTa 模型和 tokenizer
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.to(device)  # 将模型移动到 GPU
    model.eval()  # 设置为评估模式

    # 文本预处理
    input_ids, attention_masks = preprocess_texts(reasons, tokenizer, max_length)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    # 生成 RoBERTa 向量
    vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids), batch_size)):  # 使用较小的批处理
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_masks = attention_masks[i:i+batch_size]
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            # 取所有标记的隐藏状态的平均值作为句子的向量表示
            batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vectors.extend(batch_vectors)

    # 将生成的768维向量添加到原DataFrame中
    vector_df = pd.DataFrame(vectors, columns=[f'vector_{i+1}' for i in range(768)])
    df = df.drop(columns=['Reason'])  # 删除原有的'Reason'列
    df = pd.concat([df, vector_df], axis=1)  # 将向量列添加到DataFrame中

    # 输出新CSV文件
    output_file = f'csvfile/{dataset}/requirement_code_pairs_{dataset}_with_deepseekR1_vectors.csv'
    ensure_dir(output_file)
    df.to_csv(output_file, index=False)

    print(f"RoBERTa vectors have been written to {output_file}")
