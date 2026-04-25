import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        tmp_mask = torch.zeros(b, self.heads, n, n, device=x.device, requires_grad=False)
        index = torch.topk(dots, k=int(max(int(n//3), 1)), dim=-1, largest=True)[1]
        tmp_mask.scatter_(-1, index, 1.)
        attn = torch.where(tmp_mask > 0, dots, torch.full_like(dots, float('-inf')))

        attn = self.attend(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GCNAttentionEncoder(nn.Module):
    def __init__(self, net_params):
        super(GCNAttentionEncoder, self).__init__()
        self.device = net_params['device']
        self.gcn_layers = net_params['num_layers']
        self.embedding_dim = net_params['embedding_dim']
        self.num_heads = net_params['num_heads']
        self.dim_head = net_params['dim_head']
        self.dropout = net_params['dropout']
        self.residual = net_params['residual']
        self.intra_attention = net_params['intra-attention']
        self.cross_attention = net_params['cross-attention']
        # Define GCN layers
        self.gcn = nn.ModuleList()
        for _ in range(self.gcn_layers):
            self.gcn.append(GCNConv(self.embedding_dim, self.embedding_dim))

        self.self_attention_norm = nn.LayerNorm(self.embedding_dim)

        self.attnetion_norm = nn.LayerNorm(self.embedding_dim)
        self.self_attention = SelfAttention(self.embedding_dim, heads=self.num_heads, dim_head=self.dim_head)
        self.ffn_norm = nn.LayerNorm(self.embedding_dim)
        self.self_attention_dropout = nn.Dropout(self.dropout)

        self.ffn = FeedForward(self.embedding_dim, self.embedding_dim)
        self.ffn_dropout = nn.Dropout(self.dropout)

    def forward(self, x, adj):
        residuals = []

        for gcn_layer in self.gcn:
            x = gcn_layer(x, adj)
            x = torch.relu(x)
            residuals.append(x)

        if self.residual:
            gcn_result = torch.stack(residuals, dim=0).sum(dim=0)
        else:
            gcn_result = residuals[-1]

        if not self.intra_attention:
            return gcn_result.unsqueeze(0)  # Add batch dimension and return

        # Apply self-attention
        gcn_result = gcn_result.unsqueeze(0) # Add batch dimension

        attionn_result = self.self_attention_norm(gcn_result)

        attionn_result = self.self_attention(attionn_result)

        attionn_result = self.self_attention_dropout(attionn_result)

        attionn_result = attionn_result + gcn_result

        ffn_result = self.ffn_norm(attionn_result)

        ffn_result = self.ffn(ffn_result)

        ffn_result = self.ffn_dropout(ffn_result)

        self_attention_result = ffn_result + attionn_result

        result = gcn_result + self_attention_result

        return result
