class Code_graph:
    def __init__(self, x ,edge, file_name):
        self.x = x
        self.edge = edge
        self.file_name = file_name


class Queue():
    """Construct the queue structure using list
    """
    def __init__(self) -> None:
        self.__list = list()
    
    def is_empty(self):
        return self.__list == []
    
    def push(self, data):
        self.__list.append(data)
    
    def pop(self):
        if self.is_empty():
            return False
        
        return self.__list.pop(0)


class Sample:
    def __init__(self, ast_1, ast_2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                 cfg_1, cfg_2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                    dfg_1, dfg_2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2, 
                 
                 clone_label, dataset_label, clone_type, similarity_score):
        self.ast_1 = ast_1
        self.ast_2 = ast_2
        self.ast_edge_index_1 = ast_edge_index_1
        self.ast_edge_index_2 = ast_edge_index_2
        self.ast_edge_attr_1 = ast_edge_attr_1
        self.ast_edge_attr_2 = ast_edge_attr_2
        self.cfg_1 = cfg_1
        self.cfg_2 = cfg_2
        self.cfg_edge_index_1 = cfg_edge_index_1
        self.cfg_edge_index_2 = cfg_edge_index_2
        self.cfg_edge_attr_1 = cfg_edge_attr_1
        self.cfg_edge_attr_2 = cfg_edge_attr_2
        self.dfg_1 = dfg_1
        self.dfg_2 = dfg_2
        self.dfg_edge_index_1 = dfg_edge_index_1
        self.dfg_edge_index_2 = dfg_edge_index_2
        self.dfg_edge_attr_1 = dfg_edge_attr_1
        self.dfg_edge_attr_2 = dfg_edge_attr_2
        self.clone_label = clone_label
        self.dataset_label = dataset_label
        self.clone_type = clone_type
        self.similarity_score = similarity_score


class SampleEval:
    def __init__(self, index, ast, ast_edge_index, ast_edge_attr,
                 cfg, cfg_edge_index, cfg_edge_attr,
                 dfg, dfg_edge_index, dfg_edge_attr):
        self.index = index
        self.ast = ast
        self.ast_edge_index = ast_edge_index
        self.ast_edge_attr = ast_edge_attr
        self.cfg = cfg
        self.cfg_edge_index = cfg_edge_index
        self.cfg_edge_attr = cfg_edge_attr
        self.dfg = dfg
        self.dfg_edge_index = dfg_edge_index
        self.dfg_edge_attr = dfg_edge_attr


EDGE_DICT = {"ast_edge":[0], 'value_edge':[1], 'cfg_edge':[2], 'dfg_edge':[3], 'if_edge':[4], 'ifelse_edge': [5], 'while_edge':[6],
             'for_edge':[7]}
