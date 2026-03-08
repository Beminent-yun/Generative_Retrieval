import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CausalTransformer(nn.Module):
    def __init__(self, 
                 vocab_size:int,
                 num_users: int,
                 d_model:int = 128,
                 num_head:int = 4,
                 num_layers:int = 4,
                 dim_ffn:int = 512,
                 max_seq_len:int = 200,
                 dropout_rate:float = 0.1,
                 num_rq_layers:int = 3,
                 codebook_size:int = 256,
                 use_user_token: bool = True,
                 target_loss_weights: list[float] | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_rq_layers = num_rq_layers
        self.codebook_size = codebook_size
        self.use_user_token = use_user_token
        self.CODE_OFFSET = 3

        self.user_emb = None
        if self.use_user_token:
            self.user_emb = nn.Embedding(
                num_embeddings=num_users,
                embedding_dim=d_model
            )

        if target_loss_weights is None:
            weight_tensor = torch.ones(num_rq_layers, dtype=torch.float32)
        else:
            if len(target_loss_weights) != num_rq_layers:
                raise ValueError(
                    f"target_loss_weights length ({len(target_loss_weights)}) "
                    f"must equal num_rq_layers ({num_rq_layers})"
                )
            weight_tensor = torch.tensor(target_loss_weights, dtype=torch.float32)
        self.register_buffer("target_loss_weights", weight_tensor, persistent=False)
        
        # Token Embedding   token 表示
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Position Encoding  -> 区分每一个token
        self.pos_enc = PositionEncoding(d_model, max_seq_len, dropout_rate)
        # rq position encoding  -> 区分每一层
        self.rq_pos_emb = nn.Embedding(num_rq_layers, d_model)
        
        # Transformer Decoder-only (Causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_head,
            dim_feedforward=dim_ffn,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True # pre-LayerNorm, 训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出头/投影头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定 (输入emb & 输出头共享权重，减少训练参数)
        self.lm_head.weight = self.token_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _make_causal_mask(self, seq_len:int, device) -> torch.Tensor:
        """
        上三角 causal mask（bool）：True 表示被屏蔽，False 表示可见。
        与 src_key_padding_mask 保持同类型，避免不同 dtype 在部分 CUDA 版本上的数值异常。
        """
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1
        )
        

    def _make_rq_position_ids(self, seq_len:int, device) -> torch.tensor:
        """
        生成层间 position embedding/ID -> 区分层间ID/位置
        序列:      [BOS, c0_item#0, c1_item#0, c2_item#0, c0_item#1, ...]
        pos_emb:   [0,     0,          1,      2,          0, .      ...]
        """
        positions = [0] # BOS位置为0
        L = self.num_rq_layers  # codebook 个数/RQ层数
        remaining = seq_len - 1 # 除去BOS的序列长度
        for i in range(remaining):
            positions.append(i % L)
        return torch.tensor(positions[:seq_len], device=device)

    def _mask_invalid_code_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        仅保留合法 code token 的 logits，屏蔽特殊 token（PAD/BOS/EOS）以及超出 codebook 的 token。
        """
        min_token = self.CODE_OFFSET
        max_token = min(self.vocab_size, self.CODE_OFFSET + self.codebook_size)

        if min_token > 0:
            logits[..., :min_token] = float("-inf")
        if max_token < self.vocab_size:
            logits[..., max_token:] = float("-inf")
        return logits

    def forward(self, input_ids:torch.tensor,   # 输入序列: [B, T]
                attention_mask: torch.tensor,    # 掩码M: [B, T]
                user_ids: torch.Tensor | None = None  # [B,]
                )->torch.tensor:
        """
        返回 logits [B, T, ,vocab_size]
        """
        
        B, T = input_ids.shape
        device = input_ids.device
        
        # x = token_emb + pos_emb   # [B, T, D] + [B, T, D] = [B, T, D]
        x = self.pos_enc(self.token_emb(input_ids))
        
        # 加入层间位置嵌入
        rq_pos_id = self._make_rq_position_ids(T, device)   # [T, D]
        x = x + self.rq_pos_emb(rq_pos_id).unsqueeze(0) # [B, T, D] + [1, T, D]
        
        # User-Token: 把user_embedding加到第一个位置(BOS位置)
        # BOS token 同时携带了用户身份信息
        if self.use_user_token:
            if user_ids is None:
                raise ValueError("user_ids must be provided when use_user_token=True")
            user_vec = self.user_emb(user_ids)  # [B, d_model]
            x[:, 0, :] = x[:, 0, :] + user_vec
        
        attn_mask = self._make_causal_mask(T, device)
        
        # Transformer(decoder-only, 用 x 作为memory和target)
        # NOTE:
        # 在左 padding 场景下，src_key_padding_mask 在部分后端上会导致注意力行被完全屏蔽，
        # 进而产生 NaN。这里统一仅使用 causal mask，保证数值稳定。
        out = self.transformer(src=x, mask=attn_mask)
        
        logits = self.lm_head(out)  # [B, T, vocab_size]
        
        return logits
    
    def compute_loss(self,input_ids: torch.tensor,  # [B, T]
                     attention_mask:torch.tensor,   # [B, T]
                     target_ids:torch.tensor,   # [B, L] 下一个item(目标)的语义ID
                     user_ids: torch.tensor
                     )->dict:
        """
        Teacher forcing 训练损失
        训练输入拼接目标前缀：
        in_train = [history, t0, t1, ..., t(L-2)]
        监督目标 = [t0, t1, ..., t(L-1)]
        """
        L = self.num_rq_layers

        # 拼接 teacher-forcing 前缀（不包含最后一个目标 token）
        prefix_ids = target_ids[:, :-1]  # [B, L-1]
        train_input_ids = torch.cat([input_ids, prefix_ids], dim=1)  # [B, T+L-1]
        prefix_mask = torch.ones(
            target_ids.size(0), L - 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        train_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)  # [B, T+L-1]

        logits = self(train_input_ids, train_attention_mask, user_ids)    # [B, T+L-1, vocab_size]

        pred_logits = logits[:, -L:, :]    # [B, L, vocab_size]
        
        per_token_loss = F.cross_entropy(
            pred_logits.reshape(-1, self.vocab_size),   # [B*L, vocab_size]
            target_ids.reshape(-1),
            reduction='none'
        ).view(target_ids.size(0), L)

        weights = self.target_loss_weights.view(1, L)
        loss = (per_token_loss * weights).sum() / (target_ids.size(0) * weights.sum())
        
        # Compute Accuracy
        pred_ids = pred_logits.argmax(-1)   # [B, L]
        acc = (pred_ids == target_ids).all(dim=1).float().mean()
        
        return {
            "loss": loss,
            "acc": acc
        }
        