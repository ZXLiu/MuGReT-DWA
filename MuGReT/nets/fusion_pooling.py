import torch
import torch.nn as nn
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.nn import Set2Set


class FusionPoolingLayer(nn.Module):
    def __init__(self, net_params):
        super(FusionPoolingLayer, self).__init__()
        self.device = net_params['device']
        self.args_set2set = bool(net_params.get('set2set', False))
        self.embedding_dim = net_params['embedding_dim']

        self.mlp_gate = nn.Sequential(nn.Linear(self.embedding_dim,1),nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)

        self.set2set = Set2Set(self.embedding_dim, processing_steps=3)

    def _concat_present(self, *xs):
        """Concatenate non-None tensors along node dim (0). Return a single tensor."""
        present = [x for x in xs if x is not None]
        if not present:
            raise ValueError("FusionPoolingLayer: received no modality features.")
        if len(present) == 1:
            return present[0]
        return torch.cat(present, dim=0)

    def _pool_one_graph(self, x):
        """Pool node features -> graph vector (single graph)."""
        if self.args_set2set:
            return self.set2set(x)  # [1, 2*embedding_dim] (single-graph batch)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            return self.pool(x, batch=batch)  # [embedding_dim]

    def forward(self, ast1=None, ast2=None, cfg1=None, cfg2=None, dfg1=None, dfg2=None):
        """
        Early-concat behavior:
          - Build all_feat_1 = cat(enabled inputs for graph1 along dim=0)
          - Build all_feat_2 = cat(enabled inputs for graph2 along dim=0)
          - Pool once per graph with chosen method.
        """
        if self.args_set2set:
            # Early concatenation across provided modalities (per graph)
            all_feat_1 = self._concat_present(ast1, cfg1, dfg1)
            all_feat_2 = self._concat_present(ast2, cfg2, dfg2)

            # Single pooling per graph
            hg1 = self._pool_one_graph(all_feat_1)
            hg2 = self._pool_one_graph(all_feat_2)
            return hg1, hg2

        else:
            hg1_ast =  self.pool(ast1, batch=torch.zeros(ast1.size(0), dtype=torch.long, device=self.device))
            hg2_ast = self.pool(ast2, batch=torch.zeros(ast2.size(0), dtype=torch.long, device=self.device))
            hg1_cfg = self.pool(cfg1, batch=torch.zeros(cfg1.size(0), dtype=torch.long, device=self.device))  
            hg2_cfg = self.pool(cfg2, batch=torch.zeros(cfg2.size(0), dtype=torch.long, device=self.device))
            hg1_dfg = self.pool(dfg1, batch=torch.zeros(dfg1.size(0), dtype=torch.long, device=self.device))
            hg2_dfg = self.pool(dfg2, batch=torch.zeros(dfg2.size(0), dtype=torch.long, device=self.device))
            hg1 = torch.cat([hg1_ast, hg1_cfg, hg1_dfg], dim=-1)
            hg2 = torch.cat([hg2_ast, hg2_cfg, hg2_dfg], dim=-1)
            return hg1, hg2
