import os
import json
from PycharmProject.MuGReT-DWA.MuGReT.util.setting import log
from PycharmProject.MuGReT-DWA.MuGReT.data.data_tool import split_data, build_sample_eval_data
from PycharmProject.MuGReT-DWA.MuGReT.data.sast.ast_api import create_ast
from PycharmProject.MuGReT-DWA.MuGReT.data.sast.ast_api import create_ast_eval, create_ast_single
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.graph_builder import build_graph, build_graph_eval, build_graph_single


class Dataset:
    def __init__(self, dataset_params):
        self.train_data = []
        self.test_data = []
        self.new_test_data = []
        self.val_data = []
        self.vocab_length = 0 


class SemanticCodeGraphJavaDataset(Dataset):
    def __init__(self, dataset_params):
        super().__init__(dataset_params)
        dataset = dataset_params['name']
        if dataset == 'BigCloneBench':
            files_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_source/dataset_bigclonebench/')
        elif dataset == 'GoogleCodeJam':
            files_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_source/dataset_java/')
        labels_path = os.path.join(files_path, 'clone_labels.txt')
        self.skip_length = dataset_params['skip_length']

        if not os.path.exists(files_path):
            log.error(f"dataset not found!! in {files_path}")
            exit(-1)
        else:
            log.info("Creating AST...")
            save_vocab_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/VocabDict.json"
            astdict, vocabsize, vocabdict, filtered_labels, add_new_labels = create_ast(os.path.join(files_path, 'dataset_files'), labels_path, dataset, save_vocab_path)
        
        log.info("Creating separate graph...")

        graph_dict_ast, graph_dict_cfg, graph_dict_dfg, graph_dict_skip = build_graph(astdict, vocabdict, self.skip_length)

        log.info("Splitting data...")
        if not os.path.exists(labels_path):
            log.error(f"labels not found!! in {labels_path}")
            exit(-1)
        else:
            self.train_data, self.test_data, self.new_test_data, self.val_data = split_data(graph_dict_ast, graph_dict_cfg, graph_dict_dfg, filtered_labels, add_new_labels, dataset, graph_dict_skip)

        log.info("Dataset loaded successfully")
        self.vocab_length = vocabsize


class SemanticCodeGraphJavaDatasetRetrieval():
    def __init__(self, dataset_params, file_path):
        self.skip_length = dataset_params['skip_length']

        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print("Creating AST...")
        save_vocab_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/VocabDict.json"
        astdict, vocabsize, vocabdict = create_ast_eval(dataset, save_vocab_path)
    
        print("Creating separate graph...")
        graph_dict_ast, graph_dict_cfg, graph_dict_dfg, graph_dict_skip = build_graph_eval(astdict, vocabdict, self.skip_length)

        print("Splitting data...")
        self.data = build_sample_eval_data(dataset, graph_dict_ast, graph_dict_cfg, graph_dict_dfg, graph_dict_skip)

        print("Dataset loaded successfully")
        self.vocab_length = vocabsize


class SemanticCodeGraphJavaDatasetSingle():
    def __init__(self, dataset_params, data):
        self.skip_length = dataset_params['skip_length']

        print("Creating AST...")
        save_vocab_path = "/home/liutongxue/PycharmProject/MuGReT-DWA/MuGReT/VocabDict.json"
        with open(save_vocab_path, 'r', encoding='utf-8') as f:
            vocabdict = json.load(f)
        vocabsize = len(vocabdict)
        
        ast_tree = create_ast_single(data)
    
        print("Creating separate graph...")
        graph_dict_ast, graph_dict_cfg, graph_dict_dfg = build_graph_single(ast_tree, vocabdict, self.skip_length)

        self.graph_dict_ast = graph_dict_ast
        self.graph_dict_cfg = graph_dict_cfg
        self.graph_dict_dfg = graph_dict_dfg

        print("Dataset loaded successfully")
        self.vocab_length = vocabsize
