import torch
import torch.nn as nn
from einops import rearrange


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [b, h, n, d]
        b, h, n, d = x.size()

        # Global average pooling over n × d per channel (h)
        # Output: [b, h]
        y = x.mean(dim=[2, 3])  # [b, h]

        # Channel-wise weights: [b, h] → [b, h, 1, 1]
        y = self.fc(y).view(b, h, 1, 1)

        # Scale the input
        return x * y


class InterAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

        self.se_block = SEBlock(heads)

    def forward(self, x, y):
        b, nx, _ = x.size()
        b, ny, _ = y.size()        
        h = self.heads

        # q:y kv:x
        q_y = self.to_q(y)
        kv_x = self.to_kv(x).chunk(2, dim=-1)
        qkv_y = (q_y,) + kv_x
        q_y, k_x, v_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv_y)

        dots_yx = torch.einsum('b h i d, b h j d -> b h i j', q_y, k_x) * self.scale

        tmp_mask_yx = torch.zeros(b, self.heads, ny, nx, device=x.device, requires_grad=False)
        index = torch.topk(dots_yx, k=int(max(int(nx//4), 1)), dim=-1, largest=True)[1]
        tmp_mask_yx.scatter_(-1, index, 1.)
        attn_yx = torch.where(tmp_mask_yx>0, dots_yx, torch.full_like(dots_yx, float('-inf')))

        attn_yx = self.attend(attn_yx)

        out_y = torch.einsum('b h i j, b h j d -> b h i d', attn_yx, v_x)
        out_y = self.se_block(out_y)
        out_y = rearrange(out_y, 'b h n d -> b n (h d)')

        # q:x kv:y
        q_x = self.to_q(x)
        kv_y = self.to_kv(y).chunk(2, dim = -1)
        qkv_x = (q_x,) + kv_y
        q_x, k_y, v_y = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv_x)

        dots_xy = torch.einsum('b h i d, b h j d -> b h i j', q_x, k_y) * self.scale

        tmp_mask_xy = torch.zeros(b, self.heads, nx, ny, device=y.device, requires_grad=False)
        index = torch.topk(dots_xy, k=int(max(int(ny//4), 1)), dim=-1, largest=True)[1]
        tmp_mask_xy.scatter_(-1, index, 1.)
        attn_xy = torch.where(tmp_mask_xy>0, dots_xy, torch.full_like(dots_xy, float('-inf')))

        attn_xy = self.attend(attn_xy)

        out_x = torch.einsum('b h i j, b h j d -> b h i d', attn_xy, v_y)
        out_x = self.se_block(out_x)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x), self.to_out(out_y)


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, net_params):
        super(CrossAttention, self).__init__()
        self.embedding_dim = net_params['embedding_dim']
        self.num_heads = net_params['num_heads']
        self.dim_head = net_params['dim_head']
        self.cross_attention = net_params['cross-attention']
        self.fnode = torch.nn.GRUCell(self.embedding_dim, self.embedding_dim, bias=True)

        self.normalize = nn.LayerNorm(self.embedding_dim)
        self.inter_attention = InterAttention(dim=self.embedding_dim, heads=self.num_heads, dim_head=self.dim_head)
        self.ffn_norm = nn.LayerNorm(self.embedding_dim)
        self.ffn = FeedForward(self.embedding_dim, self.embedding_dim)

        self.pool = nn.AdaptiveMaxPool2d((1, self.embedding_dim))
        self.fc1 = nn.Linear(2 * self.num_heads * self.embedding_dim, 2 * self.num_heads * self.embedding_dim)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x1_embed, x2_embed):
        
        if not self.cross_attention:
            return x1_embed, x2_embed

        x1_embed = self.normalize(x1_embed)
        x2_embed = self.normalize(x2_embed)
        x1_updated, x2_updated = self.inter_attention(x1_embed, x2_embed)

        x1_updated = self.ffn_norm(x1_updated)
        x2_updated = self.ffn_norm(x2_updated)

        m1 = self.ffn(x1_updated)
        m2 = self.ffn(x2_updated)

        h1 = self.fnode(m1.squeeze(0), x1_embed.squeeze(0))

        h2 = self.fnode(m2.squeeze(0), x2_embed.squeeze(0))

        return h1, h2
