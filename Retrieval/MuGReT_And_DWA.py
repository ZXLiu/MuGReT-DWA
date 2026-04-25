import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PycharmProject.MuGReT-DWA.MuGReT.util.setting import log
from PycharmProject.MuGReT-DWA.MuGReT.nets.mugret_network import MuGReTNet
from PycharmProject.MuGReT-DWA.DWA.main import PairwiseWeightMLP, compute_weighted_similarity
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.code_graph import SampleEval
from PycharmProject.MuGReT-DWA.MuGReT.data.dataset_builder import SemanticCodeGraphJavaDatasetRetrieval
import multiprocessing as mp
import numpy as np
import json
import time
import argparse


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def execute_graph_model(process_id, model_path, defects4j_data, original_defects4j_data, retrieval_data, net_params):
    device = net_params['device']
    model = MuGReTNet(net_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Loaded model from {model_path}")

    is_main_process = (process_id == 0)

    res = []
    start_time = time.time()
    with torch.no_grad():
        for defects4j_item in tqdm(defects4j_data, desc=f"Process {process_id} | Total Progress", total=len(defects4j_data), disable=not is_main_process):
            ast_1, ast_edge_index_1, ast_edge_attr_1 = defects4j_item.ast, defects4j_item.ast_edge_index, defects4j_item.ast_edge_attr
            cfg_1, cfg_edge_index_1, cfg_edge_attr_1 = defects4j_item.cfg, defects4j_item.cfg_edge_index, defects4j_item.cfg_edge_attr
            dfg_1, dfg_edge_index_1, dfg_edge_attr_1 = defects4j_item.dfg, defects4j_item.dfg_edge_index, defects4j_item.dfg_edge_attr

            ast1 = torch.tensor(ast_1, dtype=torch.long, device=device)
            ast_edge_index_1 = torch.tensor(ast_edge_index_1, dtype=torch.long, device=device)
            ast_edge_attr_1 = torch.tensor(ast_edge_attr_1, dtype=torch.long, device=device)

            cfg1 = torch.tensor(cfg_1, dtype=torch.long, device=device)
            cfg_edge_index_1 = torch.tensor(cfg_edge_index_1, dtype=torch.long, device=device)
            cfg_edge_attr_1 = torch.tensor(cfg_edge_attr_1, dtype=torch.long, device=device)

            dfg1 = torch.tensor(dfg_1, dtype=torch.long, device=device)
            dfg_edge_index_1 = torch.tensor(dfg_edge_index_1, dtype=torch.long, device=device)
            dfg_edge_attr_1 = torch.tensor(dfg_edge_attr_1, dtype=torch.long, device=device)
            
            temp_list = []
            for retrieval_item in tqdm(retrieval_data, desc="Scanning Retrieval DB", total=len(retrieval_data), disable=not is_main_process):
                ast_2, ast_edge_index_2, ast_edge_attr_2 = retrieval_item.ast, retrieval_item.ast_edge_index, retrieval_item.ast_edge_attr
                cfg_2, cfg_edge_index_2, cfg_edge_attr_2 = retrieval_item.cfg, retrieval_item.cfg_edge_index, retrieval_item.cfg_edge_attr
                dfg_2, dfg_edge_index_2, dfg_edge_attr_2 = retrieval_item.dfg, retrieval_item.dfg_edge_index, retrieval_item.dfg_edge_attr


                ast2 = torch.tensor(ast_2, dtype=torch.long, device=device)
                ast_edge_index_2 = torch.tensor(ast_edge_index_2, dtype=torch.long, device=device)
                ast_edge_attr_2 = torch.tensor(ast_edge_attr_2, dtype=torch.long, device=device)

                cfg2 = torch.tensor(cfg_2, dtype=torch.long, device=device)
                cfg_edge_index_2 = torch.tensor(cfg_edge_index_2, dtype=torch.long, device=device)
                cfg_edge_attr_2 = torch.tensor(cfg_edge_attr_2, dtype=torch.long, device=device)

                dfg2 = torch.tensor(dfg_2, dtype=torch.long, device=device)
                dfg_edge_index_2 = torch.tensor(dfg_edge_index_2, dtype=torch.long, device=device)
                dfg_edge_attr_2 = torch.tensor(dfg_edge_attr_2, dtype=torch.long, device=device)

                data_ast = [ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2]
                data_cfg = [cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2]
                data_dfg = [dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2]
                data_input = [data_ast, data_cfg, data_dfg]

                prediction = model(data_input)
                similarity_score = F.cosine_similarity(prediction[0], prediction[1]).item()
                temp_list.append((retrieval_item.index, similarity_score))
            top5 = sorted(temp_list, key=lambda x: x[1], reverse=True)[:5]
            
            target = next(d for d in original_defects4j_data if d['index'] == defects4j_item.index)
            target['top_5_results'] = top5
            res.append(target)

    end_time = time.time()
    total = end_time - start_time
    print(f"Total evaluation time: {total:.2f} sec")

    write_json(f"/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf_normalization_results_{str(process_id)}.json", res)


def execute_dwa_model(process_id, graph_model_path, dwa_model_path, defects4j_data, original_defects4j_data, retrieval_data, net_params, defects4j_embedding, retrieval_embedding):
    device = net_params['device']
    model = MuGReTNet(net_params)
    model.load_state_dict(torch.load(graph_model_path, map_location=device))
    model.to(device)
    model.eval()

    dwa_model = PairwiseWeightMLP(256, 896)
    dwa_model.load_state_dict(torch.load(dwa_model_path, map_location=device))
    dwa_model.to(device)
    dwa_model.eval()
    print(f"Loaded graph model from {graph_model_path} and dwa model from {dwa_model_path}")

    is_main_process = (process_id == 0)

    res = []
    start_time = time.time()
    with torch.no_grad():
        for defects4j_item in tqdm(defects4j_data, desc=f"Process {process_id} | Total Progress", total=len(defects4j_data), disable=not is_main_process):
            ast_1, ast_edge_index_1, ast_edge_attr_1 = defects4j_item.ast, defects4j_item.ast_edge_index, defects4j_item.ast_edge_attr
            cfg_1, cfg_edge_index_1, cfg_edge_attr_1 = defects4j_item.cfg, defects4j_item.cfg_edge_index, defects4j_item.cfg_edge_attr
            dfg_1, dfg_edge_index_1, dfg_edge_attr_1 = defects4j_item.dfg, defects4j_item.dfg_edge_index, defects4j_item.dfg_edge_attr

            ast1 = torch.tensor(ast_1, dtype=torch.long, device=device)
            ast_edge_index_1 = torch.tensor(ast_edge_index_1, dtype=torch.long, device=device)
            ast_edge_attr_1 = torch.tensor(ast_edge_attr_1, dtype=torch.long, device=device)

            cfg1 = torch.tensor(cfg_1, dtype=torch.long, device=device)
            cfg_edge_index_1 = torch.tensor(cfg_edge_index_1, dtype=torch.long, device=device)
            cfg_edge_attr_1 = torch.tensor(cfg_edge_attr_1, dtype=torch.long, device=device)

            dfg1 = torch.tensor(dfg_1, dtype=torch.long, device=device)
            dfg_edge_index_1 = torch.tensor(dfg_edge_index_1, dtype=torch.long, device=device)
            dfg_edge_attr_1 = torch.tensor(dfg_edge_attr_1, dtype=torch.long, device=device)

            d4j_emb = [item for item in defects4j_embedding if item["index"] == defects4j_item.index][0]
            d4j_emb = torch.tensor(d4j_emb["c2llm_embedding"], device=device)
            
            temp_list = []
            for retrieval_item in tqdm(retrieval_data, desc="Scanning Retrieval DB", total=len(retrieval_data), disable=not is_main_process):
                ast_2, ast_edge_index_2, ast_edge_attr_2 = retrieval_item.ast, retrieval_item.ast_edge_index, retrieval_item.ast_edge_attr
                cfg_2, cfg_edge_index_2, cfg_edge_attr_2 = retrieval_item.cfg, retrieval_item.cfg_edge_index, retrieval_item.cfg_edge_attr
                dfg_2, dfg_edge_index_2, dfg_edge_attr_2 = retrieval_item.dfg, retrieval_item.dfg_edge_index, retrieval_item.dfg_edge_attr

                ast2 = torch.tensor(ast_2, dtype=torch.long, device=device)
                ast_edge_index_2 = torch.tensor(ast_edge_index_2, dtype=torch.long, device=device)
                ast_edge_attr_2 = torch.tensor(ast_edge_attr_2, dtype=torch.long, device=device)

                cfg2 = torch.tensor(cfg_2, dtype=torch.long, device=device)
                cfg_edge_index_2 = torch.tensor(cfg_edge_index_2, dtype=torch.long, device=device)
                cfg_edge_attr_2 = torch.tensor(cfg_edge_attr_2, dtype=torch.long, device=device)

                dfg2 = torch.tensor(dfg_2, dtype=torch.long, device=device)
                dfg_edge_index_2 = torch.tensor(dfg_edge_index_2, dtype=torch.long, device=device)
                dfg_edge_attr_2 = torch.tensor(dfg_edge_attr_2, dtype=torch.long, device=device)

                data_ast = [ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2]
                data_cfg = [cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2]
                data_dfg = [dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2]
                data_input = [data_ast, data_cfg, data_dfg]

                re_emb = [item for item in retrieval_embedding if item["index"] == retrieval_item.index][0]
                re_emb = torch.tensor(re_emb["c2llm_embedding"], device=device)

                prediction = model(data_input)

                final_score, _ = compute_weighted_similarity(dwa_model, prediction[0], d4j_emb, prediction[1], re_emb)
                similarity_score = final_score.item()

                temp_list.append((retrieval_item.index, similarity_score))
            top5 = sorted(temp_list, key=lambda x: x[1], reverse=True)[:5]
            
            target = next(d for d in original_defects4j_data if d['index'] == defects4j_item.index)
            target['top_5_results'] = top5
            res.append(target)

    end_time = time.time()
    total = end_time - start_time
    print(f"Total evaluation time: {total:.2f} sec")

    write_json(f"/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf_normalization_mlp_results_{str(process_id)}.json", res)


def execute_graph_model_parallel(graph_model_path, dwa_model_path, defects4j_data, original_defects4j_data, retrieval_data, net_params, defects4j_embedding, retrieval_embedding):
    # 1. 设置进程启动模式为 'spawn'
    # 这是原生 multiprocessing 在 GPU 环境下运行最重要的一步
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过了，就忽略报错
        pass

    num_procs = 6
    print(f"准备启动 {num_procs} 个原生进程并行计算...")

    # 2. 将数据切分为 6 份
    # 我们需要同时切分两个列表，确保 index 一一对应
    d4j_chunks = np.array_split(defects4j_data, num_procs)
    processes = []

    for i in range(num_procs):
        # 准备当前进程的参数
        # 转换为 list 确保序列化（pickle）正常
        # p_args = (
        #     i,                          # process_id
        #     graph_model_path,           # 图模型路径
        #     dwa_model_path,             # mlp模型路径
        #     d4j_chunks[i].tolist(),     # 待查数据分片
        #     original_defects4j_data,    # 原始数据分片
        #     retrieval_data,             # 完整的检索库
        #     net_params,                 # 网络参数
        #     defects4j_embedding,        # C2LLM嵌入的defects4j的向量
        #     retrieval_embedding         # C2LLM嵌入的retrieval的向量
        # )
        p_args = (
            i,                          # process_id
            graph_model_path,           # 图模型路径
            d4j_chunks[i].tolist(),     # 待查数据分片
            original_defects4j_data,    # 原始数据分片
            retrieval_data,             # 完整的检索库
            net_params                  # 网络参数
        )

        # 3. 创建并启动进程
        # target 指向你定义的单进程处理函数
        p = mp.Process(target=execute_graph_model, args=p_args)
        p.start()
        processes.append(p)
        print(f"进程 {i} 已启动 (PID: {p.pid})")

    # 4. 等待所有进程完成
    for p in processes:
        p.join()

    print("\n所有并行任务执行完毕。")
    print(f"结果已分别保存至：humaneval-java-sf_normalization_results_[0-5].json")


def main():
    parser = argparse.ArgumentParser(description='selected top n results in defects4j benchmark')
    parser.add_argument('--config', default='/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/eval/config.json')
    parser.add_argument('--retireval_graph_data_path', default='/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/eval/retireval_graph_data_human_eval_java.json')
    parser.add_argument('--human_eval_graph_data_path', default='/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/eval/human_eval_graph_data.json')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    device = "cuda"

    # model path
    graph_model_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/out/checkpoints/MuGReT/model_2.pth"
    dwa_model_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/mlp_ckpt/model_4.pth"

    # retrieval dataset
    if not os.path.exists(args.retireval_graph_data_path):
        retrieval_dataset = SemanticCodeGraphJavaDatasetRetrieval(config['dataset_params'], '/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/filter_on_top_k_and_threshold_0.3_human_eval.json')
        retrieval_data = retrieval_dataset.data

        retrieval_json_data = [obj.__dict__ for obj in retrieval_data]
        write_json(args.retireval_graph_data_path, retrieval_json_data)
    
    else:
        retrieval_data = read_json(args.retireval_graph_data_path)
        retrieval_data = [SampleEval(**item) for item in retrieval_data]
    
    # human_eval dataset
    if not os.path.exists(args.human_eval_graph_data_path):
        human_eval_dataset = SemanticCodeGraphJavaDatasetRetrieval(config['dataset_params'], '/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf_normalization.json')
        human_eval_data = human_eval_dataset.data

        human_eval_json_data = [obj.__dict__ for obj in human_eval_data]
        write_json(args.human_eval_graph_data_path, human_eval_json_data)
    
    else:
        human_eval_data = read_json(args.human_eval_graph_data_path)
        human_eval_data = [SampleEval(**item) for item in human_eval_data]

    original_human_eval_data = read_json('/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf_normalization.json')
    vocabdict = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/VocabDict.json")

    human_eval_embedding = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf_normalization_c2llm_embedding.json")
    retrieval_embedding = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/filter_on_top_k_and_threshold_0.3_human_eval_c2llm_embedding.json")
    
    # network parameters setting
    net_params = config['net_params']
    net_params['device'] = device
    net_params['vocablen'] = len(vocabdict)

    # execute
    execute_graph_model_parallel(graph_model_path=graph_model_path, dwa_model_path=dwa_model_path, defects4j_data=human_eval_data, original_defects4j_data=original_human_eval_data, retrieval_data=retrieval_data, net_params=net_params, defects4j_embedding=human_eval_embedding, retrieval_embedding=retrieval_embedding)


if __name__ == '__main__':
    main()
