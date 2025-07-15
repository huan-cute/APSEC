import os
import csv
import random
from chardet import detect
dataset = "SMOS"
# 配置路径
req_folder = f'D:/jh_code/test/dataset/uc/{dataset}'
code_folder = f'D:/jh_code/test/dataset/cc/{dataset}'
true_set_file = f'D:/jh_code/test/dataset/true_set/{dataset}.txt'
output_csv = f'D:/jh_code/test/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}.csv'  # 输出的CSV文件路径


def get_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']


def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            encoding = get_file_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except:
            for encoding in ['gbk', 'latin-1', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except:
                    continue
            return "[无法解码的文件内容]"
    except FileNotFoundError:
        return "[文件不存在]"


# 读取真集链接（保持原始文件名格式）
true_pairs = set()
with open(true_set_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            req, code = parts
            true_pairs.add((req, code))  # 保留原始文件名

# 获取所有需求文件（.txt）和代码文件（所有非.txt文件）
req_files = [f for f in os.listdir(req_folder) if f.endswith('.txt')]
code_files = [f for f in os.listdir(code_folder) if not f.endswith('.txt')]  # 获取所有非txt文件

# 创建所有可能的组合（使用原始文件名）
all_possible_pairs = []
for req in req_files:
    req_name = os.path.splitext(req)[0]  # 去掉扩展名作为req_name
    for code in code_files:
        code_name = code  # 保留完整文件名作为code_name
        all_possible_pairs.append((req_name, code_name))

# 分离真链接和非真链接
false_pairs = [pair for pair in all_possible_pairs if pair not in true_pairs]
true_pairs_list = list(true_pairs)

# 计算需要的非真链接数量 (2:1比例)
false_sample_size = min(2 * len(true_pairs_list), len(false_pairs))
selected_false_pairs = random.sample(false_pairs, false_sample_size)

# 准备所有要写入的pair（保持原始文件名）
all_pairs = []
for pair in true_pairs_list:
    all_pairs.append((pair[0], pair[1], 1))  # 真集保持原始文件名

for pair in selected_false_pairs:
    all_pairs.append((pair[0], pair[1], 0))  # 非真集也保持原始文件名

# 打乱顺序
random.shuffle(all_pairs)

# 写入CSV文件
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['req_name', 'code_name', 'req', 'code', 'label'])

    for req_name, code_name, label in all_pairs:
        # 读取需求内容（自动添加.txt后缀）
        req_file = os.path.join(req_folder, f"{req_name}.txt")
        req_content = read_file_content(req_file)

        # 读取代码内容（使用原始文件名）
        code_file = os.path.join(code_folder, code_name)
        code_content = read_file_content(code_file)

        writer.writerow([req_name, code_name, req_content, code_content, label])

print(f"CSV文件已生成: {output_csv}")
