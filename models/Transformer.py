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
        self.PAD_TOKEN = 0
        self._causal_mask_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._rq_pos_cache: dict[tuple[str, int], torch.Tensor] = {}

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
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
    
    def _cache_key(self, device: torch.device, seq_len: int) -> tuple[str, int]:
        return (str(device), seq_len)

    def _make_causal_mask(self, seq_len:int, device) -> torch.Tensor:
        """
        上三角 causal mask（bool）：True 表示被屏蔽，False 表示可见。
        与 src_key_padding_mask 保持同类型，避免不同 dtype 在部分 CUDA 版本上的数值异常。
        """
        cache_key = self._cache_key(device, seq_len)
        if cache_key not in self._causal_mask_cache:
            self._causal_mask_cache[cache_key] = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
                diagonal=1
            )
        return self._causal_mask_cache[cache_key]
        

    def _make_rq_position_ids(self, seq_len:int, device) -> torch.tensor:
        """
        生成层间 position embedding/ID -> 区分层间ID/位置
        序列:      [BOS, c0_item#0, c1_item#0, c2_item#0, c0_item#1, ...]
        pos_emb:   [0,     0,          1,      2,          0, .      ...]
        """
        cache_key = self._cache_key(device, seq_len)
        if cache_key not in self._rq_pos_cache:
            if seq_len <= 0:
                positions = torch.empty(0, dtype=torch.long, device=device)
            else:
                positions = torch.zeros(seq_len, dtype=torch.long, device=device)
                if seq_len > 1:
                    positions[1:] = torch.arange(seq_len - 1, device=device) % self.num_rq_layers
            self._rq_pos_cache[cache_key] = positions
        return self._rq_pos_cache[cache_key]

    def _compact_left_padded_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将左 padding 的输入压紧为“有效 token 左对齐、右侧 padding”的布局。
        这样 BOS 恒定在位置 0，位置编码和 padding mask 都能按真实序列工作。
        """
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids shape {input_ids.shape} must match attention_mask shape {attention_mask.shape}"
            )

        batch_size, full_len = input_ids.shape
        valid_lengths = attention_mask.sum(dim=1, dtype=torch.long)
        max_valid_len = int(valid_lengths.max().item())

        compact_ids = input_ids.new_full((batch_size, max_valid_len), self.PAD_TOKEN)
        compact_mask = attention_mask.new_zeros((batch_size, max_valid_len))

        for row, valid_len in enumerate(valid_lengths.tolist()):
            if valid_len <= 0:
                continue
            compact_ids[row, :valid_len] = input_ids[row, full_len - valid_len:full_len]
            compact_mask[row, :valid_len] = 1

        return compact_ids, compact_mask, valid_lengths

    def prepare_compact_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估/解码专用：将 left-padded 输入压紧为紧凑布局，并返回有效长度。
        """
        return self._compact_left_padded_inputs(input_ids, attention_mask)

    def _append_prefix_to_compact_inputs(
        self,
        compact_ids: torch.Tensor,
        compact_mask: torch.Tensor,
        prefix_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_lengths = compact_mask.sum(dim=1, dtype=torch.long)

        if prefix_ids is None or prefix_ids.numel() == 0:
            return compact_ids, compact_mask, valid_lengths

        if prefix_ids.dim() != 2:
            raise ValueError(f"prefix_ids must be 2D [B, S], got shape={prefix_ids.shape}")
        if prefix_ids.size(0) != compact_ids.size(0):
            raise ValueError(
                f"prefix_ids batch size ({prefix_ids.size(0)}) must match compact_ids batch size ({compact_ids.size(0)})"
            )

        prefix_len = prefix_ids.size(1)
        history_width = compact_ids.size(1)
        total_width = history_width + prefix_len

        merged_ids = compact_ids.new_full((compact_ids.size(0), total_width), self.PAD_TOKEN)
        merged_mask = compact_mask.new_zeros((compact_mask.size(0), total_width))
        merged_ids[:, :history_width] = compact_ids
        merged_mask[:, :history_width] = compact_mask

        prefix_positions = valid_lengths.unsqueeze(1) + torch.arange(prefix_len, device=compact_ids.device).unsqueeze(0)
        merged_ids.scatter_(1, prefix_positions, prefix_ids)
        merged_mask.scatter_(
            1,
            prefix_positions,
            torch.ones_like(prefix_ids, dtype=compact_mask.dtype, device=compact_mask.device)
        )

        return merged_ids, merged_mask, valid_lengths + prefix_len

    def _restore_left_padding_layout(
        self,
        compact_logits: torch.Tensor,
        full_len: int,
        valid_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        将压紧序列上的 logits 映射回原始左 padding 布局。
        这样训练和生成阶段依然可以沿用“最后几个位置就是最新 token”的取法。
        """
        batch_size = compact_logits.size(0)
        full_logits = compact_logits.new_zeros((batch_size, full_len, self.vocab_size))

        for row, valid_len in enumerate(valid_lengths.tolist()):
            if valid_len <= 0:
                continue
            full_logits[row, full_len - valid_len:full_len] = compact_logits[row, :valid_len]

        return full_logits

    def _encode_compact_inputs(
        self,
        compact_ids: torch.Tensor,
        compact_mask: torch.Tensor,
        user_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, seq_len = compact_ids.shape
        device = compact_ids.device

        x = self.pos_enc(self.token_emb(compact_ids))
        rq_pos_id = self._make_rq_position_ids(seq_len, device)
        x = x + self.rq_pos_emb(rq_pos_id).unsqueeze(0)

        if self.use_user_token:
            if user_ids is None:
                raise ValueError("user_ids must be provided when use_user_token=True")
            x[:, 0, :] = x[:, 0, :] + self.user_emb(user_ids)

        attn_mask = self._make_causal_mask(seq_len, device)
        key_padding_mask = ~compact_mask.bool()

        return self.transformer(
            src=x,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask
        )

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
        
        full_len = input_ids.size(1)
        compact_ids, compact_mask, valid_lengths = self.prepare_compact_inputs(
            input_ids, attention_mask
        )
        out = self._encode_compact_inputs(compact_ids, compact_mask, user_ids)
        compact_logits = self.lm_head(out)  # [B, T, vocab_size]
        logits = self._restore_left_padding_layout(compact_logits, full_len, valid_lengths)
        
        return logits

    def decode_last_logits(
        self,
        compact_ids: torch.Tensor,
        compact_mask: torch.Tensor,
        user_ids: torch.Tensor,
        prefix_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        评估/Beam Search 专用：在紧凑 history 后拼接 prefix，仅返回最后一个位置的 logits。
        """
        merged_ids, merged_mask, merged_lengths = self._append_prefix_to_compact_inputs(
            compact_ids, compact_mask, prefix_ids
        )
        hidden = self._encode_compact_inputs(merged_ids, merged_mask, user_ids)
        last_indices = merged_lengths - 1
        last_hidden = hidden[
            torch.arange(hidden.size(0), device=hidden.device),
            last_indices
        ]
        return self.lm_head(last_hidden)
    
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
        
