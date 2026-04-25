import torch
import torch.nn as nn
from PycharmProject.MuGReT-DWA.MuGReT.nets.cross_attention_layer import CrossAttention
from PycharmProject.MuGReT-DWA.MuGReT.nets.GCN_encoder import GCNAttentionEncoder
from PycharmProject.MuGReT-DWA.MuGReT.nets.fusion_pooling import FusionPoolingLayer


class MuGReTNet(nn.Module):
    def __init__(self, net_params):
        super(MuGReTNet, self).__init__()

        self.device = net_params['device']
        self.num_layers = net_params['num_layers']
        self.vocablen = net_params['vocablen']
        self.edgelen = net_params['edgelen']
        self.embedding_dim = net_params['embedding_dim']
        self.residual = net_params['residual']
        self.intra_attention = net_params['intra-attention']

        self.embed = nn.Embedding(self.vocablen, self.embedding_dim)

        self.GCN_embedding_layer = GCNAttentionEncoder(net_params)
        self.interaction_layer = CrossAttention(net_params)
        self.fusion_pooling_layer = FusionPoolingLayer(net_params)

    def forward(self, data):
        ast1, ast2, ast_edge_index_1, ast_edge_index_2, _, _ = data[0]
        cfg1, cfg2, cfg_edge_index_1, cfg_edge_index_2, _, _ = data[1]
        dfg1, dfg2, dfg_edge_index_1, dfg_edge_index_2, _, _ = data[2]

        ast1 = self.embed(ast1).squeeze(1)
        ast2 = self.embed(ast2).squeeze(1)
        cfg1 = self.embed(cfg1).squeeze(1)
        cfg2 = self.embed(cfg2).squeeze(1)
        dfg1 = self.embed(dfg1).squeeze(1)
        dfg2 = self.embed(dfg2).squeeze(1)

        ast1_embed = self.GCN_embedding_layer(ast1, ast_edge_index_1)
        ast2_embed = self.GCN_embedding_layer(ast2, ast_edge_index_2)
        cfg1_embed = self.GCN_embedding_layer(cfg1, cfg_edge_index_1)
        cfg2_embed = self.GCN_embedding_layer(cfg2, cfg_edge_index_2)
        dfg1_embed = self.GCN_embedding_layer(dfg1, dfg_edge_index_1)
        dfg2_embed = self.GCN_embedding_layer(dfg2, dfg_edge_index_2)

        ast1_fea, ast2_fea = self.interaction_layer(ast1_embed, ast2_embed)
        cfg1_fea, cfg2_fea = self.interaction_layer(cfg1_embed, cfg2_embed)
        dfg1_fea, dfg2_fea = self.interaction_layer(dfg1_embed, dfg2_embed)

        hg1, hg2 = self.fusion_pooling_layer(ast1=ast1_fea, ast2=ast2_fea, cfg1=cfg1_fea, cfg2=cfg2_fea, dfg1=dfg1_fea, dfg2=dfg2_fea)
        return hg1, hg2
