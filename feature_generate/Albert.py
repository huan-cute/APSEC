import os
import torch
from transformers import AlbertTokenizer, AlbertModel
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
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['smos']
    #datasets = ['Derby']
    # datasets = ['Derby', 'Drools', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2']
    # types = ['uc', 'cc']
    types = ['cc']
    max_length = 512
    batch_size = 8

    for dataset in datasets:
        for type in types:
            try:
                model_name = 'albert-base-v2'
                tokenizer = AlbertTokenizer.from_pretrained(model_name)
                model = AlbertModel.from_pretrained(model_name)
                model.to(device)
                model.eval()

                # 读取文本文件
                input_file = f'../docs/{dataset}/{type}/{type}_emb_doc.txt'
                with open(input_file, 'r', encoding='ISO-8859-1') as f:
                    lines = f.readlines()

                # 文本预处理
                input_ids, attention_masks = preprocess_texts(lines, tokenizer, max_length)
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)

                # 生成 ALBERT 向量
                vectors = []
                with torch.no_grad():
                    for i in tqdm(range(0, len(input_ids), batch_size)):
                        batch_input_ids = input_ids[i:i+batch_size]
                        batch_attention_masks = attention_masks[i:i+batch_size]
                        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
                        # 取所有标记的隐藏状态的平均值作为句子的向量表示
                        batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        vectors.extend(batch_vectors)

                # 将向量写入 Excel 文件
                df = pd.DataFrame(vectors)
                output_file = f'../docs/{dataset}/{type}/{type}_albert_vectors_2.xlsx'
                ensure_dir(output_file)
                df.to_excel(output_file, index=False)

                print(f"ALBERT vectors have been written to {output_file}")
                print(f"Sample vector shape: {vectors[0].shape}")
            except Exception as e:
                print(f"Error processing {dataset}/{type}: {e}")
