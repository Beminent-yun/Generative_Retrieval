import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_head:int, dropout_rate:float):
        super().__init__()
        
        assert d_model % num_head == 0, f"d_model({d_model}) must be divisible by num_head({num_head})!"
        
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.atten_dropout = dropout_rate
        
        # 减少一次 kernel launch 和参数管理复杂度
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.residual_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x:torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape   # [B, T, d_model]
        
        QKV = self.qkv_proj(x)  # [B, T, 3*d_model]
        Q, K, V = QKV.chunk(3, dim=-1)
        
        Q = Q.view(B, T, self.num_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = K.view(B, T, self.num_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        V = V.view(B, T, self.num_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        
        atten_mask = None   # 既不是PAD也不是未来时间步的token
        use_causal_flag = True
        
        if padding_mask is not None:
            """
            query\key   PAD PAD BOS A0 A1
            PAD          0   0   0  0  0
            PAD          0   0   0  0  0
            BOS          0   0   1  0  0
            A0           0   0   1  1  0
            A1           0   0   1  1  1

            """
            key_mask = padding_mask[:, None, None, :].bool()    # [B, 1, 1, T]
            causal_mask = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
            atten_mask = key_mask & causal_mask.view(1, 1, T, T)
            use_causal_flag = False
        
        y = F.scaled_dot_product_attention(
            query=Q, key=K, value=V,
            attn_mask=atten_mask,
            dropout_p=self.atten_dropout if self.training else 0.0,
            is_causal=use_causal_flag
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.residual_dropout(y)
        return y
    
    