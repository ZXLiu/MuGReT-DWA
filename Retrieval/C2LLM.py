import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import faiss
import numpy as np
from utils import *
from tqdm import tqdm
from transformers import AutoModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def build_or_load_index(model, corpus_data, vector_path, index_path):
    """
    自动判断并加载/构建向量库
    """
    instruction = "Represent the code snippet for retrieval: "
    
    # 检查文件是否存在
    if os.path.exists(vector_path) and os.path.exists(index_path):
        print(f"检测到现有向量库，正在从 {vector_path} 加载...")
        corpus_embeddings = np.load(vector_path)
        index = faiss.read_index(index_path)
    else:
        print("未检测到缓存文件，开始大规模并行编码 (约需要一定时间)...")
        fixed_functions = [item["fixed_function"] for item in corpus_data]
        
        # 批量编码逻辑
        all_embeddings = []
        batch_size = 128 # 根据显存大小调整
        for i in range(0, len(fixed_functions), batch_size):
            batch = [instruction + t for t in fixed_functions[i : i + batch_size]]
            with torch.no_grad():
                # 注意：某些版本的 encode 返回的是 numpy，某些是 torch tensor
                embs = model.encode(batch)
                if torch.is_tensor(embs):
                    embs = embs.float().cpu().numpy()
                all_embeddings.append(embs.astype('float32'))
            
            if i % 12800 == 0:
                print(f"已完成: {i}/{len(fixed_functions)}")

        corpus_embeddings = np.vstack(all_embeddings)
        
        # 构建 FAISS 索引 (使用内积索引 + 归一化 = 余弦相似度)
        dimension = corpus_embeddings.shape[1]
        print(f"dimention: {dimension}")
        index = faiss.IndexFlatIP(dimension)
        
        print("正在归一化并构建索引...")
        faiss.normalize_L2(corpus_embeddings)
        index.add(corpus_embeddings)
        
        # 保存到本地
        np.save(vector_path, corpus_embeddings)
        faiss.write_index(index, index_path)
        print(f"向量库已保存至: {vector_path} 和 {index_path}")

    return index, corpus_embeddings


def filter_on_top_k_and_threshold(k=400, threshold=0.3):
    model_path = "/home/liutongxue/LLM_Model/Retrieval_Model/C2LLM-0.5B"
    # 语料库路径
    corpus_json_path = "/home/liutongxue/Retrieval_Corpus/all_bug_retrieval_dataset.json" 
    # 待检索数据
    retrieval_json_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf.json"
    # 缓存文件路径
    VEC_CACHE = "/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/retrieval_embedding/corpus_emb_c2llm.npy"
    IDX_CACHE = "/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/retrieval_embedding/corpus_c2llm.index"
    
    # 1. 加载模型
    print("正在加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)

    # 2. 加载语料库以及带检索数据
    corpus_data = read_json(corpus_json_path)
    all_retrieval_data = read_json(retrieval_json_path)

    # 3. 获取索引
    index, _ = build_or_load_index(model, corpus_data, VEC_CACHE, IDX_CACHE)

    # 4. 检索
    results = []
    seen_ids = set()
    instruction = "Represent the code snippet for retrieval: "
    for project_id, project in all_retrieval_data.items():
        query_code = process_data(project["fix"])
        # 编码查询向量
        with torch.no_grad():
            query_emb = model.encode([instruction + query_code])
            if torch.is_tensor(query_emb):
                query_emb = query_emb.float().cpu().numpy()
            query_emb = query_emb.astype('float32')
        
        # 归一化并检索
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb, k)
        count = 0
        max_count = k if D[0][0] < threshold else 100
        for score, idx in zip(D[0], I[0]): 
            # 去重逻辑
            item = corpus_data[idx]
            content_hash = hash(item["buggy_function"] + item["fixed_function"])
            
            if content_hash not in seen_ids:
                # item["similarity_score"] = float(score)
                results.append(item)
                seen_ids.add(content_hash)
                count += 1
                
            # 数量达到 k 则停止（虽然 zip 遍历本身受限于 k，但这里加一步更严谨）
            if count >= max_count:
                break
    print(f"filter retrieval dataset length: {len(results)}")
    write_json("/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/filter_on_top_k_and_threshold_0.3_human_eval.json", results)


def filter_on_top_k(k=400):
    model_path = "/home/liutongxue/LLM_Model/C2LLM-0.5B"
    # 语料库路径
    corpus_json_path = "/home/liutongxue/Retrieval_Corpus/all_bug_retrieval_dataset.json" 
    # 待检索数据
    retrieval_json_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/Defects4j/single_function_repair.json"
    # 缓存文件路径
    VEC_CACHE = "/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/retrieval_embedding/corpus_emb_c2llm.npy"
    IDX_CACHE = "/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/retrieval_embedding/corpus_c2llm.index"
    
    # 1. 加载模型
    print("正在加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)

    # 2. 加载语料库以及带检索数据
    corpus_data = read_json(corpus_json_path)
    all_retrieval_data = read_json(retrieval_json_path)

    # 3. 获取索引
    index, _ = build_or_load_index(model, corpus_data, VEC_CACHE, IDX_CACHE)

    # 4. 检索
    results = []
    seen_ids = set()
    instruction = "Represent the code snippet for retrieval: "
    for project_id, project in all_retrieval_data.items():
        query_code = process_data(project["fix"])
        # 编码查询向量
        with torch.no_grad():
            query_emb = model.encode([instruction + query_code])
            if torch.is_tensor(query_emb):
                query_emb = query_emb.float().cpu().numpy()
            query_emb = query_emb.astype('float32')
        
        # 归一化并检索
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb, k)
        count = 0
        for idx in I[0]:
            count += 1
            item = corpus_data[idx]
            content_hash = hash(item["buggy_function"] + item["fixed_function"])
            
            if content_hash not in seen_ids:
                results.append(item)
                seen_ids.add(content_hash)
                
            if count >= k:
                break
    print(f"filter retrieval dataset length: {len(results)}")
    write_json("/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/filter_on_top_k.json", results)
    return results


def get_top_1():
    model_path = "/home/liutongxue/LLM_Model/Retrieval_Model/C2LLM-0.5B"
    # 语料库路径
    corpus_json_path = "/home/liutongxue/Retrieval_Corpus/all_bug_retrieval_dataset.json" 
    # 待检索数据
    retrieval_json_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf.json"
    # 缓存文件路径
    VEC_CACHE = "/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/retrieval_embedding/corpus_emb_c2llm.npy"
    IDX_CACHE = "/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/retrieval_embedding/corpus_c2llm.index"
    
    # 1. 加载模型
    print("正在加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)

    # 2. 加载语料库以及带检索数据
    corpus_data = read_json(corpus_json_path)
    all_retrieval_data = read_json(retrieval_json_path)

    # 3. 获取索引
    index, _ = build_or_load_index(model, corpus_data, VEC_CACHE, IDX_CACHE)

    # 4. 检索
    instruction = "Represent the code snippet for retrieval: "
    for project_id, project in all_retrieval_data.items():
        query_code = process_data(project["fix"])
        # 编码查询向量
        with torch.no_grad():
            query_emb = model.encode([instruction + query_code])
            if torch.is_tensor(query_emb):
                query_emb = query_emb.float().cpu().numpy()
            query_emb = query_emb.astype('float32')
        
        # 归一化并检索
        faiss.normalize_L2(query_emb)
        D, I = index.search(query_emb, 1)

        target_idx = int(I[0][0])
        item = corpus_data[target_idx]
        score = float(D[0][0])

        project["C2LLM_retrieval"] = item
        project["C2LLM_score"] = score

    write_json("/home/liutongxue/PycharmProject/MuGReT-DWA/C2LLM/top_1_human_eval.json", all_retrieval_data)


def get_and_save_embedding(dataset, save_embedding_path):
    model_path = "/home/liutongxue/LLM_Model/Retrieval_Model/C2LLM-0.5B"
    
    print("正在加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)

    res = []
    instruction = "Represent the code snippet for retrieval: "
    for data in tqdm(dataset, total=len(dataset), desc="Process dataset..."):
        with torch.no_grad():
            emb = model.encode([instruction + data["fixed_function"]])
            if torch.is_tensor(emb):
                emb = emb.float().cpu().tolist()
            res.append({"index": data["index"], "c2llm_embedding": emb})
    write_json(save_embedding_path, res)


def load_save_embedding(save_embedding_path, index):
    all_embedding_data = read_json(save_embedding_path)
    emb_item = [item for item in all_embedding_data if item["index"] == index][0]
    return torch.tensor(emb_item["c2llm_embedding"])


if __name__ == '__main__':
    # filter_on_top_k_and_threshold(k=300, threshold=0.3)
    get_top_1()
    
    # defects4j_dataset = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/Defects4j/single_function_repair_normalization.json")
    # save_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/Defects4j/single_function_repair_normalization_c2llm_embedding.json"
    # get_and_save_embedding(defects4j_dataset, save_path)

    # human_eval_dataset = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf_normalization.json")
    # save_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/HumanEval/humaneval-java-sf_normalization_c2llm_embedding.json"
    # get_and_save_embedding(human_eval_dataset, save_path)

    # retrieval_dataset = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/filter_retrieval_dataset/filter_on_top_k_and_threshold_0.3_human_eval.json")
    # save_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/filter_retrieval_dataset/filter_on_top_k_and_threshold_0.3_human_eval_c2llm_embedding.json"
    # get_and_save_embedding(retrieval_dataset, save_path)
