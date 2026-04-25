import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from utils import *
from tqdm import tqdm
from PycharmProject.MuGReT-DWA.MuGReT.data.dataset_builder import SemanticCodeGraphJavaDatasetSingle


forbideen_files = ['37044', '4892654', '6966398', '7550876']


def get_and_save_gcj_data(folder_path, label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip().split(',') for label in labels]
    
    code_files = []
    for ele in labels:
        code_files.append(ele[0])
        code_files.append(ele[1])
        code_files.append(ele[-2])

    code_files = sorted(set(code_files))

    res = {}
    error_index_list = []
    config = read_json("/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/config.json")
    for code_file in tqdm(code_files, total=len(code_files), desc="Processing gcj data..."):
        index = code_file
        file_name = index + '.java'
        try:
            code = read_file(os.path.join(folder_path, file_name))
            graph_object = SemanticCodeGraphJavaDatasetSingle(config["dataset_params"], code)
            ast, ast_edge_index, ast_edge_attr = graph_object.graph_dict_ast[0]
            cfg, cfg_edge_index, cfg_edge_attr = graph_object.graph_dict_cfg[0]
            dfg, dfg_edge_index, dfg_edge_attr = graph_object.graph_dict_dfg[0]
            temp_dict = {
                "code": code,
                "ast": ast, "ast_edge_index": ast_edge_index, "ast_edge_attr": ast_edge_attr,
                "cfg": cfg, "cfg_edge_index": cfg_edge_index, "cfg_edge_attr": cfg_edge_attr,
                "dfg": dfg, "dfg_edge_index": dfg_edge_index, "dfg_edge_attr": dfg_edge_attr
            }
            res[index] = temp_dict
        except Exception as e:
            error_index_list.append(index)
            continue
            
    print(f"res length: {len(res)}")
    write_json("/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/gcj_data.json", res)
    print(f"error index list length: {len(error_index_list)}")
    write_file("/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/gcj_error_index_list.txt", "\n".join(error_index_list))


def get_labels(label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip().split(',') for label in labels]

    filtered_labels = []
    for ele in labels:
        dataset_lable = int(ele[4])
        if dataset_lable == 1:  # GCJ data
            if str(ele[1]) in forbideen_files or str(ele[0]) in forbideen_files:
                continue
            filtered_labels.append(ele)
    return filtered_labels


def filter_gcj_labels(label_path, data_path):
    gcj_data = read_json(data_path)
    right_index_list = list(gcj_data.keys())

    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip().split(',') for label in labels]

    filtered_labels = []
    for ele in labels:
        if ele[0] in right_index_list and ele[1] in right_index_list and ele[-2] in right_index_list:
            filtered_labels.append(",".join(ele))
    print(f"before filter gcj length: {len(labels)}, after filter gcj length: {len(filtered_labels)}")
    write_file("/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/gcj_label_dataset.json", "\n".join(filtered_labels))


def build_train_data():
    new_labels = []
    labels = get_labels("/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/data/data_source/dataset_bigclonebench/clone_labels.txt")
    
    non_clones_list = [item for item in labels if item[3] == '0' and item[2] == '0']  # 训练集且是非克隆
    clones_list = [item for item in labels if item[3] == '0' and item[2] == '1']  # 训练集且是克隆

    for label in tqdm(clones_list, total=len(clones_list), desc="process gcj clone labels..."):
        index_1 = label[0]
        index_1_list = [item for item in non_clones_list if item[0] == index_1]
        if len(index_1_list) == 0: continue
        random_item = random.choice(index_1_list)
        label.append(random_item[1])  # 随机找一个index_1的非克隆并取出他的index
        label.append(random_item[2])  # 随机非克隆的标签(0表示非克隆)
        new_labels.append(label)
    print(f"new labels length: {len(new_labels)}")

    random.shuffle(new_labels)
    new_labels = [",".join(item) for item in new_labels]
    write_file("/home/liutongxue/PycharmProject/MuGReT-DWA/DWA/gcj_label_dataset.json", "\n".join(new_labels))


if __name__ == '__main__':
    build_train_data()
