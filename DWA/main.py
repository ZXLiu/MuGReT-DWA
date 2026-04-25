import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PycharmProject.MuGReT-DWA.MuGReT.nets.mugret_network import MuGReTNet
from utils import *
from tqdm import tqdm, trange


class PairwiseWeightMLP(nn.Module):
    def __init__(self, g_dim=256, t_dim=896):
        super(PairwiseWeightMLP, self).__init__()
        input_dim = (g_dim + t_dim) * 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 alpha，用于加权文本相似度
        )

    def forward(self, g1, t1, g2, t2):
        diff_g = torch.abs(g1 - g2)
        diff_t = torch.abs(t1 - t2)

        mult_g = g1 * g2
        mult_t = t1 * t2

        combined_features = torch.cat([diff_g, diff_t, mult_g, mult_t], dim=-1)
        return self.net(combined_features)


def compute_weighted_similarity(mlp_model, g1, t1, g2, t2):
    """
    输入两对特征，输出动态加权后的相似度分数
    """
    # 计算基础相似度分数 (ST, SG)
    st = F.cosine_similarity(t1, t2, dim=-1).unsqueeze(-1)
    sg = F.cosine_similarity(g1, g2, dim=-1).unsqueeze(-1)
    
    # 计算特征绝对差异，并输入 MLP 得到权重 a
    a = mlp_model(g1, t1, g2, t2)
    
    # 按照公式融合: S = a * ST + (1 - a) * SG
    final_s = a * st + (1.0 - a) * sg
    return final_s, a


def train():
    root_ckpt_dir = "/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/mlp_ckpt"
    
    labels_dataset_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/gcj_label_dataset.json"
    with open(labels_dataset_path, 'r') as f:
        labels_dataset = f.readlines()
        labels_dataset = [label_item.strip().split(',') for label_item in labels_dataset]
    
    dataset = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/gcj_data.json")
    
    config = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/config.json")
    device = "cuda"
    vocabdict = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/VocabDict.json")

    # graph model parameters setting
    net_params = config['net_params']
    net_params['device'] = device
    net_params['vocablen'] = len(vocabdict)

    # 训练参数
    batch_size = 32
    g_dim, t_dim = 256, 896
    lr = 1e-4
    epoch_nums = 10

    def create_batches(data):
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return batches

    # 初始化 MLP model以及graph model
    graph_model_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/out/checkpoints/MuGReT/model_2.pth"
    model = MuGReTNet(net_params)
    model.load_state_dict(torch.load(graph_model_path, map_location=device))
    model.to(device)
    model.train()
    for param in model.parameters():  # # 显式冻结 graph model 的所有参数
        param.requires_grad = False
    print(f"Loaded graph model from {graph_model_path}")

    mlp = PairwiseWeightMLP(g_dim, t_dim).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    mlp.train()  # 只训练mlp的参数

    print("开始训练...")
    epochs = trange(epoch_nums, leave=True, desc="Epoch")
    for epoch in epochs:
        batches = create_batches(labels_dataset)
        total_loss = 0.0
        main_index = 0.0
        epoch_start = time.time()
        for batch in tqdm(batches, total=len(batches), desc="Batches"):
            optimizer.zero_grad()
            batchloss = 0
            for data in batch:
                index_a, index_p, index_n = data[0], data[1], data[-2]
                ast_a, ast_edge_index_a, ast_edge_attr_a = dataset[index_a]["ast"], dataset[index_a]["ast_edge_index"], dataset[index_a]["ast_edge_attr"]
                cfg_a, cfg_edge_index_a, cfg_edge_attr_a = dataset[index_a]["cfg"], dataset[index_a]["cfg_edge_index"], dataset[index_a]["cfg_edge_attr"]
                dfg_a, dfg_edge_index_a, dfg_edge_attr_a = dataset[index_a]["dfg"], dataset[index_a]["dfg_edge_index"], dataset[index_a]["dfg_edge_attr"]
                
                ast_a = torch.tensor(ast_a, dtype=torch.long, device=device)
                ast_edge_index_a = torch.tensor(ast_edge_index_a, dtype=torch.long, device=device)
                ast_edge_attr_a = torch.tensor(ast_edge_attr_a, dtype=torch.long, device=device)

                cfg_a = torch.tensor(cfg_a, dtype=torch.long, device=device)
                cfg_edge_index_a = torch.tensor(cfg_edge_index_a, dtype=torch.long, device=device)
                cfg_edge_attr_a = torch.tensor(cfg_edge_attr_a, dtype=torch.long, device=device)

                dfg_a = torch.tensor(dfg_a, dtype=torch.long, device=device)
                dfg_edge_index_a = torch.tensor(dfg_edge_index_a, dtype=torch.long, device=device)
                dfg_edge_attr_a = torch.tensor(dfg_edge_attr_a, dtype=torch.long, device=device)

                ast_p, ast_edge_index_p, ast_edge_attr_p = dataset[index_p]["ast"], dataset[index_p]["ast_edge_index"], dataset[index_p]["ast_edge_attr"]
                cfg_p, cfg_edge_index_p, cfg_edge_attr_p = dataset[index_p]["cfg"], dataset[index_p]["cfg_edge_index"], dataset[index_p]["cfg_edge_attr"]
                dfg_p, dfg_edge_index_p, dfg_edge_attr_p = dataset[index_p]["dfg"], dataset[index_p]["dfg_edge_index"], dataset[index_p]["dfg_edge_attr"]
                
                ast_p = torch.tensor(ast_p, dtype=torch.long, device=device)
                ast_edge_index_p = torch.tensor(ast_edge_index_p, dtype=torch.long, device=device)
                ast_edge_attr_p = torch.tensor(ast_edge_attr_p, dtype=torch.long, device=device)

                cfg_p = torch.tensor(cfg_p, dtype=torch.long, device=device)
                cfg_edge_index_p = torch.tensor(cfg_edge_index_p, dtype=torch.long, device=device)
                cfg_edge_attr_p = torch.tensor(cfg_edge_attr_p, dtype=torch.long, device=device)

                dfg_p = torch.tensor(dfg_p, dtype=torch.long, device=device)
                dfg_edge_index_p = torch.tensor(dfg_edge_index_p, dtype=torch.long, device=device)
                dfg_edge_attr_p = torch.tensor(dfg_edge_attr_p, dtype=torch.long, device=device)

                ast_n, ast_edge_index_n, ast_edge_attr_n = dataset[index_n]["ast"], dataset[index_n]["ast_edge_index"], dataset[index_n]["ast_edge_attr"]
                cfg_n, cfg_edge_index_n, cfg_edge_attr_n = dataset[index_n]["cfg"], dataset[index_n]["cfg_edge_index"], dataset[index_n]["cfg_edge_attr"]
                dfg_n, dfg_edge_index_n, dfg_edge_attr_n = dataset[index_n]["dfg"], dataset[index_n]["dfg_edge_index"], dataset[index_n]["dfg_edge_attr"]
                
                ast_n = torch.tensor(ast_n, dtype=torch.long, device=device)
                ast_edge_index_n = torch.tensor(ast_edge_index_n, dtype=torch.long, device=device)
                ast_edge_attr_n = torch.tensor(ast_edge_attr_n, dtype=torch.long, device=device)

                cfg_n = torch.tensor(cfg_n, dtype=torch.long, device=device)
                cfg_edge_index_n = torch.tensor(cfg_edge_index_n, dtype=torch.long, device=device)
                cfg_edge_attr_n = torch.tensor(cfg_edge_attr_n, dtype=torch.long, device=device)

                dfg_n = torch.tensor(dfg_n, dtype=torch.long, device=device)
                dfg_edge_index_n = torch.tensor(dfg_edge_index_n, dtype=torch.long, device=device)
                dfg_edge_attr_n = torch.tensor(dfg_edge_attr_n, dtype=torch.long, device=device)
                
                data_ast_ap = [ast_a, ast_p, ast_edge_index_a, ast_edge_index_p, ast_edge_attr_a, ast_edge_attr_p]
                data_cfg_ap = [cfg_a, cfg_p, cfg_edge_index_a, cfg_edge_index_p, cfg_edge_attr_a, cfg_edge_attr_p]
                data_dfg_ap = [dfg_a, dfg_p, dfg_edge_index_a, dfg_edge_index_p, dfg_edge_attr_a, dfg_edge_attr_p]
                data_input_ap = [data_ast_ap, data_cfg_ap, data_dfg_ap]

                data_ast_an = [ast_a, ast_n, ast_edge_index_a, ast_edge_index_n, ast_edge_attr_a, ast_edge_attr_n]
                data_cfg_an = [cfg_a, cfg_n, cfg_edge_index_a, cfg_edge_index_n, cfg_edge_attr_a, cfg_edge_attr_n]
                data_dfg_an = [dfg_a, dfg_n, dfg_edge_index_a, dfg_edge_index_n, dfg_edge_attr_a, dfg_edge_attr_n]
                data_input_an = [data_ast_an, data_cfg_an, data_dfg_an]
                
                with torch.no_grad():  # 使用graph model只用于推理作用，不训练
                    prediction_ap = model(data_input_ap)
                    prediction_an = model(data_input_an)

                t_a = torch.tensor(dataset[index_a]["c2llm_embedding"], device=device)
                t_p = torch.tensor(dataset[index_p]["c2llm_embedding"], device=device)
                t_n = torch.tensor(dataset[index_n]["c2llm_embedding"], device=device)

                # 计算正样本对相似度
                s_pos, _ = compute_weighted_similarity(mlp, prediction_ap[0], t_a, prediction_ap[1], t_p)
                # 计算负样本对相似度
                s_neg, _ = compute_weighted_similarity(mlp, prediction_an[0], t_a, prediction_an[1], t_n)

                batchloss += (2.0 - (s_pos - s_neg)).mean()
            
            batchloss.backward()
            optimizer.step()
            total_loss += batchloss.item()
            main_index += len(batch)
            loss = total_loss / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
        
        epoch_end = time.time()
        print(f"Saving model in epoch: {epoch}")
        torch.save(mlp.state_dict(), f"{root_ckpt_dir}/model_{epoch}.pth")
        print(f"Epoch {epoch} finished in {epoch_end - epoch_start:.2f} sec")


if __name__ == '__main__':
    train()
