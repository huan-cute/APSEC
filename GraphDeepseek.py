import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv  # PyG中的异构图注意力卷积模块
from torch_geometric.data import HeteroData  # 存储异构图数据
import torch_geometric.transforms as T  # 图结构转换模块
from torch_geometric.loader import LinkNeighborLoader  # 边预测任务的采样器
import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# 替换 HAN 类为 HGT 类
class HGTModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, metadata, heads=8, num_layers=2):
        super(HGTModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # HGT 通常包含多层 HGTConv
        for i in range(num_layers):
            conv = HGTConv(
                in_channels if i == 0 else hidden_channels,  # 输入维度为前一层的输出，第一层用原始特征输入
                hidden_channels,
                metadata,
                heads=heads,  # 将 'num_heads' 改为 'heads'
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_channels))  # HGTConv 输出为 (hidden_channels * heads)

        # 用于分类  ？？？？？？？？
        # 平均池化
        self.pool = nn.AvgPool1d(kernel_size=2)  # 对每个节点的特征做平均池化
        self.bn_2 = nn.BatchNorm1d(hidden_channels // 2)
        # 全连接层
        self.fc1 = nn.Linear(hidden_channels // 2, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)  # 输出层，假设是二分类任务

    def forward(self, x_dict, edge_index_dict):
        # 图卷积部分
        for conv, bn in zip(self.convs, self.bns):
            x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict.keys():
                x = x_dict[key]
                x = bn(x)
                x = F.relu(x)
                x_dict[key] = x
        # 对每种节点类型(req和code)分别进行全连接网络处理
        for key in x_dict.keys():
            x = x_dict[key]

            # 池化操作
            x = x.unsqueeze(1)  # 扩展维度以适应1D池化,将 (batch_size, num_features) 变为 (batch_size, 1, num_features) 以便应用 1D 池化
            x = self.pool(x)
            x = x.squeeze(1)  # 去除多余维度，变回 (batch_size, num_features/2)

            # 批量归一化和激活
            x = self.bn_2(x)
            x = F.relu(x)

            # 全连接层及后续批量归一化和激活
            x = self.fc1(x)
            x = self.bn3(x)
            x = F.relu(x)

            x = self.fc2(x)
            x = self.bn4(x)
            x = F.relu(x)

            x = self.fc3(x)
            x = self.bn5(x)
            x = F.relu(x)

            x = self.fc4(x)
            x = self.bn6(x)
            x = F.relu(x)

            x = self.fc5(x)

            x_dict[key] = x

        return x_dict


# 计算两个节点间的相似得分(点积+sigmoid)
class Classifier(torch.nn.Module):  # 链接预测分类器
    def forward(self, x_req: Tensor, x_code: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_req = x_req[edge_label_index[0]]  # 获取七点(需求)的嵌入
        edge_feat_code = x_code[edge_label_index[1]]  # 获取终点(代码)的嵌入
        return torch.sigmoid((edge_feat_req * edge_feat_code).sum(dim=-1))  # 点积+sigmoid作为概率


# 总模型包装(HGT+分类器)
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super(Model, self).__init__()
        self.hgt = HGTModel(in_channels, out_channels, metadata)
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "req": data["req"].x,
            "code": data["code"].x,
            "reason": data["reason"].x
        }
        x_dict = {k: v for k, v in x_dict.items() if v is not None}
        x_dict = self.hgt(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["req"],
            x_dict["code"],
            data["req", "link", "code"].edge_label_index,
        )
        return pred


def generate_reason_edges(dataset_name, reason_csv_path):
    """生成Reason相关边和节点

    Args:
        dataset_name: 数据集名称（如'iTrust'）
        reason_csv_path: 包含reason向量的CSV文件路径
    """
    # 读取CSV文件
    reason_df = pd.read_csv(reason_csv_path)

    # 创建索引字典
    uc_dir = f'D:/jh_code/test/dataset/uc/{dataset_name}'
    cc_dir = f'D:/jh_code/test/dataset/cc/{dataset_name}'

    uc_names = [f.split('.')[0] for f in os.listdir(uc_dir) if f.endswith('.txt')]
    cc_names = [f.split('.')[0] for f in os.listdir(cc_dir) if f.endswith(('.java', '.jsp'))]

    uc_idx_dict = {name: i for i, name in enumerate(uc_names)}
    cc_idx_dict = {name: j for j, name in enumerate(cc_names)}  # 代码节点索引从需求节点之后开始

    # 初始化存储
    req_treason_edge_from = []
    req_treason_edge_to = []
    treason_code_edge_from = []
    treason_code_edge_to = []
    req_freason_edge_from = []
    req_freason_edge_to = []
    freason_code_edge_from = []
    freason_code_edge_to = []

    # 创建reason索引字典
    treason_idx_dict = {}  # 存储正例reason的索引
    freason_idx_dict = {}  # 存储负例reason的索引
    t_idx = 0
    f_idx = 0
    for _, row in reason_df.iterrows():
        req_name = row['req_name']
        code_name = row['code_name'].split('.')[0]  # 移除后缀
        label = row['label']
        req_idx = uc_idx_dict[req_name]
        code_idx = cc_idx_dict[code_name]

        # 创建唯一键（组合需求名和代码名）
        reason_key = f"{req_name}_{code_name}"

        if label == 1:  # 正例
            treason_idx_dict[reason_key] = t_idx
            req_treason_edge_from.append(req_idx)
            req_treason_edge_to.append(t_idx)
            treason_code_edge_from.append(t_idx)
            treason_code_edge_to.append(code_idx)
            t_idx = t_idx + 1
        else:  # 负例
            # 分配新索引（从正例之后开始）
            freason_idx_dict[reason_key] = f_idx
            req_freason_edge_from.append(req_idx)
            req_freason_edge_to.append(t_idx)
            freason_code_edge_from.append(t_idx)
            freason_code_edge_to.append(code_idx)
            f_idx = f_idx + 1
    return req_treason_edge_from, req_treason_edge_to, treason_code_edge_from, treason_code_edge_to, req_freason_edge_from, req_freason_edge_to, freason_code_edge_from, freason_code_edge_to

def generate_req_code_edge(dataset_name):
    uc_names = os.listdir('D:/jh_code/test/dataset/' + 'uc/' + dataset_name)
    cc_names = os.listdir('D:/jh_code/test/dataset/' + 'cc/' + dataset_name)
    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    with open('D:/jh_code/test/dataset/true_set/' + f"{dataset_name}.txt", 'r', encoding='ISO8859-1') as df:
        lines = df.readlines()
    for line in lines:
        uc_name, cc_name = line.split(' ')[0], line.split(' ')[1].split('.')[0]
        uc_name = f"{uc_name}.txt"
        if uc_name in uc_idx_dict and cc_name in cc_idx_dict:
            edge_from.append(uc_idx_dict[uc_name])
            edge_to.append(cc_idx_dict[cc_name])
    return edge_from, edge_to


# 从 Excel 中提取继承边（Class 1 ->extend-> Class 2）
def generate_extend_edges(dataset_name):
    cc_names = os.listdir('D:/jh_code/test/dataset/cc/' + dataset_name)
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    extend_df = pd.read_excel(f'D:/jh_code/test/docs/{dataset_name}/cc/{dataset_name}ClassRelationships_version2.xlsx')
    for _, row in extend_df.iterrows():
        cc_name1, cc_name2, relationship = row['Class 1'], row['Class 2'], row['Relationship']
        if cc_name1 in cc_idx_dict and cc_name2 in cc_idx_dict:
            if relationship == 'extend':
                edge_from.append(cc_idx_dict[cc_name1])
                edge_to.append(cc_idx_dict[cc_name2])
    return edge_from, edge_to


# 生成调用边，生成导入边（Class 1 ->import-> Class 2）
def generate_import_edges(dataset_name):
    cc_names = os.listdir('D:/jh_code/test/dataset/cc/' + dataset_name)
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}
    edge_from, edge_to = [], []
    extend_df = pd.read_excel(f'D:/jh_code/test/docs/{dataset_name}/cc/{dataset_name}ClassRelationships_version2.xlsx')
    for _, row in extend_df.iterrows():
        cc_name1, cc_name2, relationship = row['Class 1'], row['Class 2'], row['Relationship']
        if cc_name1 in cc_idx_dict and cc_name2 in cc_idx_dict:
            if relationship == 'import':
                edge_from.append(cc_idx_dict[cc_name1])
                edge_to.append(cc_idx_dict[cc_name2])
    return edge_from, edge_to


# 使用多种IR特征计算平均排名，选择排名前50%的需求-代码边
def generate_IR_edges(dataset_name):
    # 获取需求和代码的文件名列表
    uc_names = os.listdir(f'D:/jh_code/test/dataset/uc/{dataset_name}')
    cc_names = os.listdir(f'D:/jh_code/test/dataset/cc/{dataset_name}')

    # 创建索引字典
    uc_idx_dict = {uc_names[i]: i for i in range(len(uc_names))}
    cc_idx_dict = {cc_names[j].split('.')[0]: j for j in range(len(cc_names))}

    # 读取相似度数据
    data = pd.read_excel(f'D:/jh_code/test/docs/{dataset_name}/IR_feature.xlsx')

    # 对每一列进行排名
    ranked_data = data[['vsm_1', 'vsm_2', 'lsi_1', 'lsi_2', 'lda_1', 'lda_2', 'bm25_1', 'bm25_2', 'JS_1', 'JS_2']].rank(
        method='average')

    # 计算平均排名
    ranked_data['avg_rank'] = ranked_data.mean(axis=1)
    # 组合 code, requirement 和 avg_rank 列
    combined_data = pd.DataFrame({
        'requirement': data['requirement'],
        'code': data['code'],
        'avg_rank': ranked_data['avg_rank']
    })
    # 选择前50%的链接
    top_10_percent_threshold = combined_data['avg_rank'].quantile(0.5)
    top_links = combined_data[combined_data['avg_rank'] <= top_10_percent_threshold]

    # 生成边关系
    edge_from = [uc_idx_dict[requirement] for requirement in top_links['requirement']]
    edge_to = [cc_idx_dict[code] for code in top_links['code']]

    return edge_from, edge_to


if __name__ == '__main__':
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 定义要遍历的数据集和节点特征
    datasets = ['maven']
    # datasets = ['Infinispan', 'iTrust', 'Maven', 'Pig', 'Seam2','Derby', 'Drools','Groovy', 'smos', 'Dronology']

    # 指定需求文档（UC）节点的特征类型（如 BERT, RoBERTa 等）
    # uc_nodes_features = ['bert', 'albert', 'roberta', 'xlnet']
    uc_nodes_features = ['roberta']

    # 指定代码节点（CC）的特征类型（如 CodeBERT, GraphCodeBERT 等）
    cc_nodes_features = ['graphcodebert']
    # cc_nodes_features = ['codebert', 'codet5', 'graphcodebert']

    prompt_nodes_features = ['roberta']

    # 定义要遍历的边类型组合，定义不同的边类型组合，用于构建异构图结构
    edge_type_combinations = [
            ('None',),  # 仅包含基础 link 边
            # ('IR',),  # 加入 IR 相似性边
            # ('extend',),  # 加入继承边
            # ('import',),  # 加入 import 边
            # ('IR', 'extend'),  # 同时加入 IR 与 extend
            # ('IR', 'import'),  # 同时加入 IR 与 import
            # ('import', 'extend'),  # 同时加入 import 与 extend
            # ('IR', 'extend', 'import'),
            # ('IR', 'extend', 'import','prompt'),# 所有辅助边都加入
            ('prompt')

        ]

    # 准备一个总的结果列表
    all_results = []

    # 遍历所有特征组合与数据集
    for uc_nodes_feature in uc_nodes_features:
        for cc_nodes_feature in cc_nodes_features:
            for dataset in datasets:
                # 加载 UC 和 CC 节点的预训练特征向量
                uc_df = pd.read_excel(f'D:/jh_code/test/docs/{dataset}/uc/uc_{uc_nodes_feature}_vectors.xlsx')
                cc_df = pd.read_excel(f'D:/jh_code/test/docs/{dataset}/cc/cc_{cc_nodes_feature}_vectors.xlsx')
                # reason_csv_path = f"D:/jh_code/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}_with_vectors.csv"
                reason_csv_path = f"D:/jh_code/test/test/test1/csvfile/{dataset}/requirement_code_pairs_{dataset}_with_deepseekR1_vectors.csv"
                reason_df = pd.read_csv(reason_csv_path)
                req_feat = torch.from_numpy(uc_df.values).to(torch.float)
                code_feat = torch.from_numpy(cc_df.values).to(torch.float)
                reason_feat = torch.from_numpy(reason_df.iloc[:, 3:].values).to(torch.float)
                print(reason_df.columns)  # 查看所有列名
                print(reason_df.head())  # 查看前几行数据
                print(torch.unique(reason_feat))  # 查看特征值范围
                # 加载基础 link 边（需求到代码的真实链接）
                edge_from, edge_to = generate_req_code_edge(dataset)
                edge_index = torch.tensor([edge_from, edge_to], dtype=torch.long)
                # 调用方式
                req_treason_from, req_treason_to, treason_code_from, treason_code_to, req_freason_from, req_freason_to, freason_code_from, freason_code_to = generate_reason_edges(dataset, reason_csv_path)

                # 生成所有可能的边，加载其他辅助边类型
                # extend_edge_from, extend_edge_to = generate_extend_edges(dataset)
                # import_edge_from, import_edge_to = generate_import_edges(dataset)
                # IR_edge_from, IR_edge_to = generate_IR_edges(dataset)
                # extend_edge_index = torch.tensor([extend_edge_from, extend_edge_to], dtype=torch.long)
                # import_edge_index = torch.tensor([import_edge_from, import_edge_to], dtype=torch.long)
                # IR_edge_index = torch.tensor([IR_edge_from, IR_edge_to], dtype=torch.long)
                req_treason_index = torch.tensor([req_treason_from, req_treason_to], dtype=torch.long)
                treason_code_index = torch.tensor([treason_code_from, treason_code_to], dtype=torch.long)
                req_freason_index = torch.tensor([req_freason_from, req_freason_to], dtype=torch.long)
                freason_code_index = torch.tensor([freason_code_from, freason_code_to], dtype=torch.long)

                # 遍历边类型组合，遍历所有边类型组合配置
                for edge_types in edge_type_combinations:
                    # 初始化HeteroData
                    data = HeteroData()
                    data["req"].x = req_feat  # 需求节点特征
                    data["code"].x = code_feat  # 代码节点特征
                    data["reason"].x = reason_feat
                    data["req", "link", "code"].edge_index = edge_index  # 真实链接边

                    # 根据当前组合添加边，动态添加辅助边
                    # if 'extend' in edge_types:
                    #     data["code", "extend", "code"].edge_index = extend_edge_index
                    # if 'import' in edge_types:
                    #     data["code", "import", "code"].edge_index = import_edge_index
                    # if 'IR' in edge_types:
                    #     data["req", "sim", "code"].edge_index = IR_edge_index
                    if 'prompt' in edge_types:
                        data["req", "req_treason", "reason"].edge_index = req_treason_index
                        data["reason", "treason_code", "code"].edge_index = treason_code_index
                        data["req", "req_freason", "reason"].edge_index = req_freason_index
                        data["reason", "freason_code", "code"].edge_index = freason_code_index
                    # 遍历所有边类型组合配置
                    data = T.ToUndirected()(data)
                    # 初始化分数列表 初始化每次实验的结果存储列表
                    precision_scores = []
                    recall_scores = []
                    f1_scores = []

                    # 对每种配置进行多次随机实验（10次）
                    for i in range(50):
                        print(
                            f"Dataset: {dataset}, Node Feature: {uc_nodes_feature}&{cc_nodes_feature}, Edge Types: {edge_types}, Experiment {i + 1}/50")
                        # 使用 PyG 的 RandomLinkSplit 对链接边进行训练-测试划分
                        transform = T.RandomLinkSplit(
                            num_test=0.1,  # 测试集比例为10%
                            disjoint_train_ratio=0.3,  # 训练集断开比例（防止数据泄露）
                            neg_sampling_ratio=2.0,  # 负样本比例；l
                            add_negative_train_samples=False,
                            edge_types=("req", "link", "code"),
                        )

                        train_data, _, test_data = transform(data)  # 忽略验证集
                        # 构建训练集加载器，按邻居采样
                        train_loader = LinkNeighborLoader(
                            data=train_data,
                            num_neighbors=[20, 10],  # 每层采样邻居数
                            neg_sampling_ratio=2.0,
                            edge_label_index=(
                            ("req", "link", "code"), train_data["req", "link", "code"].edge_label_index),
                            edge_label=train_data["req", "link", "code"].edge_label,
                            batch_size=128,
                            shuffle=True,
                        )

                        # 初始化模型
                        model = Model(in_channels=req_feat.size(1), out_channels=128, metadata=data.metadata())
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        # 训练模型(共训练30轮)

                        for epoch in range(1, 30):
                            total_loss = total_examples = 0
                            for sampled_data in train_loader:
                                try:
                                    optimizer.zero_grad()
                                    sampled_data = sampled_data.to(device)
                                    pred = model(sampled_data)
                                    ground_truth = sampled_data["req", "link", "code"].edge_label
                                    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                                    loss.backward()
                                    optimizer.step()
                                    total_loss += float(loss) * pred.numel()
                                    total_examples += pred.numel()
                                except IndexError as e:
                                    print(f"IndexError: {e}")
                                    print(f"Sampled Data: {sampled_data}")
                                    break
                        # 构建测试加载器
                        test_loader = LinkNeighborLoader(
                            data=test_data,
                            num_neighbors=[20, 10],
                            edge_label_index=(
                            ("req", "link", "code"), test_data["req", "link", "code"].edge_label_index),
                            edge_label=test_data["req", "link", "code"].edge_label,
                            batch_size=3 * 128,
                            shuffle=False,
                        )
                        # 在测试集上评估性能
                        preds = []
                        ground_truths = []
                        for sampled_data in test_loader:
                            with torch.no_grad():
                                sampled_data = sampled_data.to(device)
                                preds.append(model(sampled_data).cpu())
                                ground_truths.append(sampled_data["req", "link", "code"].edge_label.cpu())
                        pred = torch.cat(preds, dim=0).numpy()
                        ground_truth = torch.cat(ground_truths, dim=0).numpy()
                        pred_labels = (pred > 0.5).astype(np.float32)

                        precision = precision_score(ground_truth, pred_labels, average='binary')
                        recall = recall_score(ground_truth, pred_labels, average='binary')
                        f1 = f1_score(ground_truth, pred_labels, average='binary')

                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        f1_scores.append(f1)

                    # 计算平均值
                    avg_precision = np.mean(precision_scores)
                    avg_recall = np.mean(recall_scores)
                    avg_f1 = np.mean(f1_scores)

                    print(f"Average Precision: {avg_precision:.4f}")
                    print(f"Average Recall: {avg_recall:.4f}")
                    print(f"Average F1: {avg_f1:.4f}")

                    # 记录结果
                    all_results.append({
                        'Dataset': dataset,
                        'UC Node Feature': uc_nodes_feature,
                        'CC Node Feature': cc_nodes_feature,
                        'Edge Types': '+'.join(edge_types),
                        'Precision': avg_precision,
                        'Recall': avg_recall,
                        'F1': avg_f1
                    })

    # 将所有结果写入Excel
    results_df = pd.DataFrame(all_results)
    results_df.to_excel('D:\jh_code\\test\\test\\test1\\result\hgt_rq1_with_reason_edges.xlsx', index=False)
    print("All experiments completed. Results saved to './result1/hgt_rq1_with_reason_edges.xlsx'.")