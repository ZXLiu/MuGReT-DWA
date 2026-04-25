import os
from PycharmProject.MuGReT-DWA.MuGReT.util.setting import log
from javalang.ast import Node
from anytree import AnyNode, RenderTree
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.ast_builder import get_ast_edge, get_value_edge, get_ast_edge_without_value_edge
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.cfg_builder import get_cfg_edge
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.fa_builder import get_if_edge, get_loops_edge, get_next_sib_edge, get_next_use_edge, get_next_token_edge
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.dfg_builder import get_dfg_edge
from PycharmProject.MuGReT-DWA.MuGReT.data.sast.ast_api import createtree
from PycharmProject.MuGReT-DWA.MuGReT.data.graph_builder.code_graph import EDGE_DICT
from tqdm import tqdm


def build_graph(astdict, vocabdict, skip_length):
    graph_dict_ast = {}
    graph_dict_cfg = {}
    graph_dict_dfg = {}
    graph_dict_skip = []
    
    for path, tree in tqdm(astdict.items()):
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None, is_statement=False)
        createtree(newtree, tree, nodelist)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr = []

        get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)

        ast_x = x
        ast_edge_index = [edgesrc, edgetgt]
        ast_edge_attr = edge_attr

        if skip_length and len(ast_x) <= skip_length:
            # build cfg and dfg edge
            get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)
            
            newtree, edgesrc, edgetgt, edge_attr = get_dfg_edge(newtree, edgesrc, edgetgt, edge_attr)

            edgesrc, edgetgt, edge_attr = remove_duplicates_edges(edgesrc, edgetgt, edge_attr)

            cfg_x, new_cfg_edge, cfg_edge_attr, dfg_x, new_dfg_edge, dfg_edge_attr = split_cfg_dfg(x, edgesrc, edgetgt, edge_attr)

            file_name = os.path.splitext(os.path.basename(path))[0]

            ast_edge_attr = [inner[0] for inner in ast_edge_attr]
            graph_dict_ast[file_name] = [[ast_x, ast_edge_index, ast_edge_attr], len(ast_x)]

            cfg_edge_attr = [inner[0] for inner in cfg_edge_attr]
            graph_dict_cfg[file_name] = [[cfg_x, new_cfg_edge, cfg_edge_attr], len(cfg_x)]

            dfg_edge_attr = [inner[0] for inner in dfg_edge_attr]
            graph_dict_dfg[file_name] = [[dfg_x, new_dfg_edge, dfg_edge_attr], len(dfg_x)]

        else:
            file_name = os.path.splitext(os.path.basename(path))[0]
            graph_dict_skip.append(file_name)

    return graph_dict_ast, graph_dict_cfg, graph_dict_dfg, graph_dict_skip


def build_graph_eval(astdict, vocabdict, skip_length):
    graph_dict_ast = {}
    graph_dict_cfg = {}
    graph_dict_dfg = {}
    graph_dict_skip = []
    
    for index, tree in tqdm(astdict.items()):
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None, is_statement=False)
        createtree(newtree, tree, nodelist)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr = []

        try:
            get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
        except Exception as e:
            graph_dict_skip.append(index)
            continue

        ast_x = x
        ast_edge_index = [edgesrc, edgetgt]
        ast_edge_attr = edge_attr

        if skip_length and len(ast_x) <= skip_length:
            # build cfg and dfg edge
            get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)
            
            newtree, edgesrc, edgetgt, edge_attr = get_dfg_edge(newtree, edgesrc, edgetgt, edge_attr)

            edgesrc, edgetgt, edge_attr = remove_duplicates_edges(edgesrc, edgetgt, edge_attr)

            cfg_x, new_cfg_edge, cfg_edge_attr, dfg_x, new_dfg_edge, dfg_edge_attr = split_cfg_dfg(x, edgesrc, edgetgt, edge_attr)

            ast_edge_attr = [inner[0] for inner in ast_edge_attr]
            graph_dict_ast[index] = [[ast_x, ast_edge_index, ast_edge_attr], len(ast_x)]

            cfg_edge_attr = [inner[0] for inner in cfg_edge_attr]
            graph_dict_cfg[index] = [[cfg_x, new_cfg_edge, cfg_edge_attr], len(cfg_x)]

            dfg_edge_attr = [inner[0] for inner in dfg_edge_attr]
            graph_dict_dfg[index] = [[dfg_x, new_dfg_edge, dfg_edge_attr], len(dfg_x)]

        else:
            graph_dict_skip.append(index)
    print(f"skip count: {len(graph_dict_skip)}")
    return graph_dict_ast, graph_dict_cfg, graph_dict_dfg, graph_dict_skip


def build_graph_single(ast, vocabdict, skip_length):
    if not ast: return [], [], []
    nodelist = []
    newtree = AnyNode(id=0, token=None, data=None, is_statement=False)
    createtree(newtree, ast, nodelist)
    x = []
    edgesrc = []
    edgetgt = []
    edge_attr = []

    try:
        get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
    except Exception as e:
        print(e)
        return [], [], []

    ast_x = x
    ast_edge_index = [edgesrc, edgetgt]
    ast_edge_attr = edge_attr

    if skip_length and len(ast_x) <= skip_length:
        # build cfg and dfg edge
        get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)

        newtree, edgesrc, edgetgt, edge_attr = get_dfg_edge(newtree, edgesrc, edgetgt, edge_attr)

        edgesrc, edgetgt, edge_attr = remove_duplicates_edges(edgesrc, edgetgt, edge_attr)

        cfg_x, new_cfg_edge, cfg_edge_attr, dfg_x, new_dfg_edge, dfg_edge_attr = split_cfg_dfg(x, edgesrc, edgetgt, edge_attr)

        ast_edge_attr = [inner[0] for inner in ast_edge_attr]
        graph_dict_ast = [[ast_x, ast_edge_index, ast_edge_attr], len(ast_x)]

        cfg_edge_attr = [inner[0] for inner in cfg_edge_attr]
        graph_dict_cfg = [[cfg_x, new_cfg_edge, cfg_edge_attr], len(cfg_x)]

        dfg_edge_attr = [inner[0] for inner in dfg_edge_attr]
        graph_dict_dfg = [[dfg_x, new_dfg_edge, dfg_edge_attr], len(dfg_x)]
        
        return graph_dict_ast, graph_dict_cfg, graph_dict_dfg
    else:
        return [], [], []


def split_cfg_dfg(x, edgesrc, edgetgt, edge_attr):
    cfg_edge = []
    cfg_edge_attr = []
    dfg_edge = []
    dfg_edge_attr = []

    for src, tgt, attr in zip(edgesrc, edgetgt, edge_attr):
        if attr[0] == EDGE_DICT['cfg_edge']:
            cfg_edge.append((src, tgt))
            cfg_edge_attr.append(attr)
        elif attr[0] == EDGE_DICT['dfg_edge']:
            dfg_edge.append((src, tgt))
            dfg_edge_attr.append(attr)

    cfg_nodes = set()
    dfg_nodes = set()

    for src, tgt in cfg_edge:
        cfg_nodes.add(src)
        cfg_nodes.add(tgt)
    for src, tgt in dfg_edge:
        dfg_nodes.add(src)
        dfg_nodes.add(tgt)

    cfg_x = [x[i] for i in cfg_nodes]
    dfg_x = [x[i] for i in dfg_nodes]
    
    cfg_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(cfg_nodes))}
    dfg_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(dfg_nodes))}

    new_cfg_edge_tuples = [(cfg_index_map[src], cfg_index_map[tgt]) for src, tgt in cfg_edge]
    new_dfg_edge_tuples = [(dfg_index_map[src], dfg_index_map[tgt]) for src, tgt in dfg_edge]

    new_cfg_edge = [[], []]
    for src, tgt in new_cfg_edge_tuples:
        new_cfg_edge[0].append(src)
        new_cfg_edge[1].append(tgt)
    
    new_dfg_edge = [[], []]
    for src, tgt in new_dfg_edge_tuples:
        new_dfg_edge[0].append(src)
        new_dfg_edge[1].append(tgt)

    if not cfg_x:
        cfg_x = [[0]]  # Add a dummy node
    if not dfg_x:
        dfg_x = [[0]]  # Add a dummy node

    if not new_cfg_edge[0]:
        new_cfg_edge = [[0], [0]]
    if not new_dfg_edge[0]:
        new_dfg_edge = [[0], [0]]
    
    if not cfg_edge_attr:
        cfg_edge_attr = [[0]]
    
    if not dfg_edge_attr:
        dfg_edge_attr = [[0]]
    
    return cfg_x, new_cfg_edge, cfg_edge_attr, dfg_x, new_dfg_edge, dfg_edge_attr


def delete_edge(edgesrc, edgetgt, edge_attr, edge_type):
    deleted_elements = [
        (src, tgt, attr)
        for src, tgt, attr in zip(edgesrc, edgetgt, edge_attr)
        if attr == edge_type
    ]
    filtered = [
        (src, tgt, attr)
        for src, tgt, attr in zip(edgesrc, edgetgt, edge_attr)
        if attr != edge_type
    ]

    # Convert the filtered results to lists
    edgesrc, edgetgt, edge_attr = (
        list(x) for x in zip(*filtered)
    ) if filtered else ([], [], [])

    return edgesrc, edgetgt, edge_attr


def remove_duplicates_edges(edgesrc, edgetgt, edge_attr):
    """
    Removes duplicate elements from the given edges based on (edgesrc, edgetgt, edge_attr) tuples.
    """
    seen = set()
    unique_edges = []
    for src, tgt, attr in zip(edgesrc, edgetgt, edge_attr):
        attr_value = attr[0][0]
        edge = (src, tgt, attr_value)
        if edge not in seen:
            unique_edges.append((src, tgt, attr))
            seen.add(edge)
    if unique_edges:
        edgesrc, edgetgt, edge_attr = zip(*unique_edges)
    else:
        edgesrc, edgetgt, edge_attr = [], [], []
    return list(edgesrc), list(edgetgt), list(edge_attr)
