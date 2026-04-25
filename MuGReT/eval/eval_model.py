import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PycharmProject.MuGReT-DWA.MuGReT.util.setting import log
from PycharmProject.MuGReT-DWA.MuGReT.nets.mugret_network import MuGReTNet
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PycharmProject.MuGReT-DWA.MuGReT.eval.load_eval import bcb_evaluation
from PycharmProject.MuGReT-DWA.MuGReT.data.dataset_builder import SemanticCodeGraphJavaDataset
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.code_graph import Sample
from transformers import AutoTokenizer, AutoModel
import json
import time
import argparse


class Data:
    def __init__(self, file_name_1, file_name_2, clone_label, clone_type, similarity_score):
        self.file_name_1 = file_name_1
        self.file_name_2 = file_name_2
        self.clone_label = clone_label
        self.clone_type = clone_type
        self.similarity_score = similarity_score


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def transfer_label(label):
    if label == 0:
        return -1
    else:
        return 1


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device(f"cuda:{str(gpu_id)}")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def print_model_parameter_weights(model_path):
    state_dict = torch.load(model_path, map_location='cuda:1')

    print(f"总共有 {len(state_dict)} 个参数张量（包括权重和偏置）\n")

    # 打印每一层的名称和对应的形状
    for layer_name, weights in state_dict.items():
        print(f"层名: {layer_name:<40} | 形状: {list(weights.shape)}")


def filter_bcb_test_data():
    forbidden_files = ['37044', '4892654', '6966398', '7550876']
    label_path = "/data/lzx/PycharmProject/MuGReT-DWA/MuGReT/data/data_source/dataset_bigclonebench/clone_labels.txt"
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip().split(',') for label in labels]

    filtered_labels = []
    for ele in labels:
        dataset_lable = int(ele[4])
        dataset_type = int(ele[3])
        if dataset_lable == 0 and dataset_type == 1:
            if str(ele[1]) in forbidden_files or str(ele[0]) in forbidden_files:
                continue
            filtered_labels.append(ele)
    return filtered_labels


def evaluate_graph_model(model_path, dataset, net_params):
    device = net_params['device']
    model = MuGReTNet(net_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Loaded model from {model_path}")

    bcb_samples = []
    results = []
    start_time = time.time()
    with torch.no_grad():
        for data in tqdm(dataset, desc="Processing Dataset", total=len(dataset)):
            ast_1, ast_2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2 = data.ast_1, data.ast_2, data.ast_edge_index_1, data.ast_edge_index_2, data.ast_edge_attr_1, data.ast_edge_attr_2
            cfg_1, cfg_2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2 = data.cfg_1, data.cfg_2, data.cfg_edge_index_1, data.cfg_edge_index_2, data.cfg_edge_attr_1, data.cfg_edge_attr_2
            dfg_1, dfg_2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2 = data.dfg_1, data.dfg_2, data.dfg_edge_index_1, data.dfg_edge_index_2, data.dfg_edge_attr_1, data.dfg_edge_attr_2
            label = data.clone_label
            label = transfer_label(label)
            label = torch.tensor(label, dtype=torch.float, device=device)

            ast1 = torch.tensor(ast_1, dtype=torch.long, device=device)
            ast2 = torch.tensor(ast_2, dtype=torch.long, device=device)
            ast_edge_index_1 = torch.tensor(ast_edge_index_1, dtype=torch.long, device=device)
            ast_edge_index_2 = torch.tensor(ast_edge_index_2, dtype=torch.long, device=device)
            ast_edge_attr_1 = torch.tensor(ast_edge_attr_1, dtype=torch.long, device=device)
            ast_edge_attr_2 = torch.tensor(ast_edge_attr_2, dtype=torch.long, device=device)

            cfg1 = torch.tensor(cfg_1, dtype=torch.long, device=device)
            cfg2 = torch.tensor(cfg_2, dtype=torch.long, device=device)
            cfg_edge_index_1 = torch.tensor(cfg_edge_index_1, dtype=torch.long, device=device)
            cfg_edge_index_2 = torch.tensor(cfg_edge_index_2, dtype=torch.long, device=device)
            cfg_edge_attr_1 = torch.tensor(cfg_edge_attr_1, dtype=torch.long, device=device)
            cfg_edge_attr_2 = torch.tensor(cfg_edge_attr_2, dtype=torch.long, device=device)

            dfg1 = torch.tensor(dfg_1, dtype=torch.long, device=device)
            dfg2 = torch.tensor(dfg_2, dtype=torch.long, device=device)
            dfg_edge_index_1 = torch.tensor(dfg_edge_index_1, dtype=torch.long, device=device)
            dfg_edge_index_2 = torch.tensor(dfg_edge_index_2, dtype=torch.long, device=device)
            dfg_edge_attr_1 = torch.tensor(dfg_edge_attr_1, dtype=torch.long, device=device)
            dfg_edge_attr_2 = torch.tensor(dfg_edge_attr_2, dtype=torch.long, device=device)

            data_ast = [ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2]
            data_cfg = [cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2]
            data_dfg = [dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2]
            data_input = [data_ast, data_cfg, data_dfg]

            prediction = model(data_input)
            output = F.cosine_similarity(prediction[0], prediction[1])
            data.similarity_score = output.item()
            results.append(data)

            if output.item() > 0:  # threshold
                prediction = int(1)
            else:
                prediction = int(0)

            bcb_samples.append((data, prediction))

    bcb_evaluation(bcb_samples)

    end_time = time.time()
    total = end_time - start_time
    print(f"Total evaluation time: {total:.2f} sec")
    print("Evaluation completed")

    with open('/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/eval/GraphModel_bcb_result.json', 'w', encoding='utf-8') as f:
        json_data = [obj.__dict__ for obj in results]
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def evaluate_normal_model(model_path, dataset, device):
    dir_path = "/data/lzx/PycharmProject/MuGReT-DWA/MuGReT/data/data_source/dataset_bigclonebench/dataset_files"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    bcb_samples = []
    results = []
    start_time = time.time()
    for ele in tqdm(dataset, desc="Processing Dataset", total=len(dataset)):
        file_name_1 = ele[0] + '.java'
        file_name_2 = ele[1] + '.java'
        file_path_1 = os.path.join(dir_path, file_name_1)
        file_path_2 = os.path.join(dir_path, file_name_2)

        code_snippet_1 = read_file(file_path_1)
        code_snippet_2 = read_file(file_path_2)

        with torch.no_grad():
            code_embedding_1 = model(**tokenizer(code_snippet_1, truncation=True, padding=True, return_tensors='pt').to(device))
            code_embedding_1 = code_embedding_1.last_hidden_state[:, 0, :]
            code_embedding_1 = code_embedding_1.cpu().detach()

            code_embedding_2 = model(**tokenizer(code_snippet_2, truncation=True, padding=True, return_tensors='pt').to(device))
            code_embedding_2 = code_embedding_2.last_hidden_state[:, 0, :]
            code_embedding_2 = code_embedding_2.cpu().detach()
        output = F.cosine_similarity(code_embedding_1, code_embedding_2)
        data = Data(file_name_1=file_name_1, file_name_2=file_name_2, clone_label=int(ele[2]), clone_type=str(ele[5]), similarity_score=output.item())
        results.append(data)

        if output.item() > 0.8:  # threshold
            prediction = int(1)
        else:
            prediction = int(0)

        bcb_samples.append((data, prediction))

    bcb_evaluation(bcb_samples)

    end_time = time.time()
    total = end_time - start_time
    print(f"Total evaluation time: {total:.2f} sec")
    print("Evaluation completed")

    with open('/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/eval/NormalModel_bcb_result.json', 'w', encoding='utf-8') as f:
        json_data = [obj.__dict__ for obj in results]
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description='evaluate models')
    parser.add_argument('--model_type', default='graph_model')
    parser.add_argument('--config', default='/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/configs/bcb.json')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    # network parameters setting
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = config['params']['batch_size']

    # model path
    if args.model_type == 'graph_model':
        model_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/out/checkpoints/MuGReT/model_2.pth"
    else:
        model_path = "/home/liutongxue/LLM_Model/Retrieval_Model/C2LLM-0.5B"

    # dataset
    if args.model_type == 'graph_model':
        dataset = SemanticCodeGraphJavaDataset(config['dataset_params'])
        net_params['vocablen'] = dataset.vocab_length
        test_data = dataset.test_data
        # test_data = dataset.new_test_data
    else:
        test_data = filter_bcb_test_data()

    # evaluate
    if args.model_type == 'graph_model':
        evaluate_graph_model(model_path, test_data, net_params)
    else:
        evaluate_normal_model(model_path, test_data, device)


if __name__ == '__main__':
    main()
