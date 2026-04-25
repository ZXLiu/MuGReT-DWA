from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.code_graph import Sample, SampleEval
from PycharmProject.MuGReT-DWA.MuGReT.util.setting import log


def split_data(graph_dict_ast, graph_dict_cfg, graph_dict_dfg, labels: list, add_new_labels: list, dataset_name: str, graph_dict_skip):
    train_data = []
    test_data, new_test_data = [], []
    val_data = []
    total_skip_length = 0

    for ele in labels:
        file_name_1 = ele[0]
        file_name_2 = ele[1]
        clone_label = int(ele[2])
        split_label = int(ele[3])
        dataset_label = int(ele[4])
        clone_type = str(ele[5])
        similarity_score = float(ele[6])
        if file_name_1 in graph_dict_skip or file_name_2 in graph_dict_skip:
            total_skip_length += 1
            continue
        else:
            if split_label == 0:
                if dataset_name=='BigCloneBench' and dataset_label==0:
                    ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                    ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                    cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                    cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                    dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                    dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                    train_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                            cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                            dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                            clone_label, dataset_label, clone_type, similarity_score))

                if dataset_name == 'GoogleCodeJam' and dataset_label==1:
                    ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                    ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                    cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                    cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                    dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                    dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                    train_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                            cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                            dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                            clone_label, dataset_label, clone_type, similarity_score))
            elif split_label == 1:
                if dataset_name=='BigCloneBench' and dataset_label==0:
                    ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                    ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                    cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                    cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                    dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                    dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                    test_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                                cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                                dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                                clone_label, dataset_label, clone_type, similarity_score))
                if dataset_name == 'GoogleCodeJam' and dataset_label==1:
                    ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                    ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                    cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                    cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                    dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                    dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                    test_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                                cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                                dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                                clone_label, dataset_label, clone_type, similarity_score))
            elif split_label == 2:
                if dataset_name=='BigCloneBench' and dataset_label==0:
                    ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                    ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                    cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                    cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                    dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                    dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                    val_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                                cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                                dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                                clone_label, dataset_label, clone_type, similarity_score))
                if dataset_name == 'GoogleCodeJam' and dataset_label==1:
                    ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                    ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                    cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                    cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                    dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                    dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                    val_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                                cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                                dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                                clone_label, dataset_label, clone_type, similarity_score))

    for new_ele in add_new_labels:
        file_name_1 = new_ele[0]
        file_name_2 = new_ele[1]
        clone_label = int(new_ele[2])
        split_label = int(new_ele[3])
        dataset_label = int(new_ele[4])
        clone_type = str(new_ele[5])
        similarity_score = float(new_ele[6])
        if file_name_1 in graph_dict_skip or file_name_2 in graph_dict_skip:
            total_skip_length += 1
            continue
        else:
            if split_label == 0:
                ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                train_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                        cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                        dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                        clone_label, dataset_label, clone_type, similarity_score))
            elif split_label == 1:
                ast1, ast_edge_index_1, ast_edge_attr_1 = graph_dict_ast[file_name_1][0]
                ast2, ast_edge_index_2, ast_edge_attr_2 = graph_dict_ast[file_name_2][0]
                cfg1, cfg_edge_index_1, cfg_edge_attr_1 = graph_dict_cfg[file_name_1][0]
                cfg2, cfg_edge_index_2, cfg_edge_attr_2 = graph_dict_cfg[file_name_2][0]
                dfg1, dfg_edge_index_1, dfg_edge_attr_1 = graph_dict_dfg[file_name_1][0]
                dfg2, dfg_edge_index_2, dfg_edge_attr_2 = graph_dict_dfg[file_name_2][0]
                new_test_data.append(Sample(ast1, ast2, ast_edge_index_1, ast_edge_index_2, ast_edge_attr_1, ast_edge_attr_2,
                                            cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, cfg_edge_attr_1, cfg_edge_attr_2,
                                            dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, dfg_edge_attr_1, dfg_edge_attr_2,
                                            clone_label, dataset_label, clone_type, similarity_score))
    log.info(f"Total skip length: {total_skip_length}")
    return train_data, test_data, new_test_data, val_data


def build_sample_eval_data(dataset, graph_dict_ast, graph_dict_cfg, graph_dict_dfg, graph_dict_skip):
    data = []
    total_skip_length = 0

    for item in dataset:
        index = item["index"]
        if index in graph_dict_skip:
            total_skip_length += 1
            continue
        else:
            ast, ast_edge_index, ast_edge_attr = graph_dict_ast[index][0]
            cfg, cfg_edge_index, cfg_edge_attr = graph_dict_cfg[index][0]
            dfg, dfg_edge_index, dfg_edge_attr = graph_dict_dfg[index][0]
            data.append(SampleEval(index, ast, ast_edge_index, ast_edge_attr,
                                    cfg, cfg_edge_index, cfg_edge_attr,
                                    dfg, dfg_edge_index, dfg_edge_attr))

    print(f"Total skip length: {total_skip_length}")
    return data
