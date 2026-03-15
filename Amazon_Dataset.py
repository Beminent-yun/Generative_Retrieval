import pickle
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = 0   # 填充，attention 时忽略
BOS_TOKEN = 1   # 序列开始
EOS_TOKEN = 2   # 序列结束
CODE_OFFSET = 3 # RQ-VAE code的起始token ID : code 0 → token 3, code 255 → token 258


def code2token(code:int) -> int:
    """
    RQ-VAE 的 code 转换成 token ID
    """
    return code + CODE_OFFSET

def token2code(token:int) -> int:
    """
    token ID 转换为 RQ-VAE的code
    """
    return token - CODE_OFFSET


class ItemEmbeddingDataset(Dataset):
    """
    RQ-VAE 训练使用
    Input:
        - 所有item构成的embedding矩阵 (Extracted by Sentence-BERT)
    Output:
        - Single item embedding vector
    
    -为什么这么简单：
      RQ-VAE 的训练目标是重建 embedding
      不需要用户信息，不需要序列结构
      就是普通的自编码器训练
    """
    def __init__(self, embeddings: np.ndarray):
        # embeddings: [num_items, embedding_dim]
        # 存为float32,减少内存占用
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
    def __len__(self)->int:
        return len(self.embeddings)
    
    def __getitem__(self, idx:int)->torch.tensor:
        return self.embeddings[idx] # Look up [embedding_dim,]
    

class SeqTrainDataset(Dataset):
    """
    序列推荐模型训练用
    每条样本代表一个 (用户历史行为序列 -> 目标item) 的预测目标
    
    target_ids:
     目标item的3个 SID token (target ids 不在input_ids里)
     [c0_target, c1_target, c2_target]
     
    Note:
        val 和 test 阶段的历史行为序列只用train部分，不能将val_target加入到历史序列（数据泄露）
        
    Sliding Window Augmentation:
        每个用户的训练历史 → 多个训练样本
        每个样本用不同长度的历史子序列预测下一个 item

        例子（window_size=5，min_len=2）：
        用户历史 train = [i1, i2, i3, i4, i5, i6]

        样本1：history=[i1, i2],       target=i3
        样本2：history=[i1, i2, i3],   target=i4
        样本3：history=[i1, i2, i3, i4], target=i5
        样本4：history=[i2, i3, i4, i5], target=i6  ← 超过 window_size，滑动
    """
    def __init__(self, user_histories: Dict[int, List[int]],    # {user_id:[item_id,...]}
                 targets: Dict[int, int],   # {user_id:target_item_id}
                 semantic_ids: np.ndarray,  # Look up table (num_items, num_rq_layers)
                 max_seq_len:int = 50,  # 用户历史行为序列的最大长度（item数）
                 num_rq_layers:int = 3,     # 每个item对应几个token/code
                 use_sliding_window: bool = True,
                 sliding_window_mode: str = "all",
                 window_size:int = 20,  # 历史窗口最大长度
                 min_seq_len: int = 2,  # 历史序列的最小长度
                 windows_per_user: int = 2,
                 seed: int = 42,
                 ):    
        if semantic_ids.ndim != 2:
            raise ValueError(f"semantic_ids must be 2D [num_items, num_rq_layers], got shape={semantic_ids.shape}")

        sid_layers = semantic_ids.shape[1]
        if num_rq_layers != sid_layers:
            raise ValueError(
                f"num_rq_layers ({num_rq_layers}) does not match semantic_ids second dim ({sid_layers})"
            )

        if min_seq_len < 1:
            raise ValueError(f"min_seq_len must be >= 1, got {min_seq_len}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if sliding_window_mode not in {"all", "sample_per_epoch"}:
            raise ValueError(
                f"sliding_window_mode must be 'all' or 'sample_per_epoch', got {sliding_window_mode}"
            )
        if windows_per_user < 1:
            raise ValueError(f"windows_per_user must be >= 1, got {windows_per_user}")

        self.user_histories = user_histories
        self.targets = targets
        self.semantic_ids = semantic_ids
        self.max_seq_len = max_seq_len
        self.num_rq_layers = num_rq_layers
        
        # max_tokens: BOS + max_seq_len * num_rq_layers(1id->num_rq_layers token)
        self.max_tokens = 1 + max_seq_len * num_rq_layers
        
        self.use_sliding_window = use_sliding_window
        self.sliding_window_mode = sliding_window_mode
        self.window_size = window_size
        self.min_seq_len = min_seq_len
        self.windows_per_user = windows_per_user
        self.seed = seed
        
        # 构建样本列表
        self.samples: List[tuple[int, List[int], int]] = []
        self.resample_samples(epoch=0)

    def _build_window_candidates(self, user_id: int, history: List[int]) -> List[tuple[int, List[int], int]]:
        candidates = []
        for end in range(self.min_seq_len, len(history)):
            sub_hist = history[max(0, end - self.window_size):end]
            sub_target = history[end]
            candidates.append((user_id, sub_hist, sub_target))
        return candidates
        
    def _build_samples(self, rng: np.random.RandomState | None = None):
        """
        构建所有训练样本
        
        use_sliding_window=True: 训练增强，多样本
        use_sliding_window=False: 每用户单样本
        """
        samples = []

        for user_id, target in self.targets.items():
            history = self.user_histories.get(user_id)
            if history is None or len(history) == 0:
                continue

            if self.use_sliding_window:
                if self.sliding_window_mode == "all":
                    # 历史太短：退化成单样本
                    if len(history) < self.min_seq_len:
                        samples.append((user_id, history[-self.window_size:], target))
                        continue

                    samples.extend(self._build_window_candidates(user_id, history))
                else:
                    window_candidates = self._build_window_candidates(user_id, history)
                    if len(window_candidates) <= self.windows_per_user:
                        samples.extend(window_candidates)
                    elif rng is not None:
                        chosen_idx = rng.choice(
                            len(window_candidates),
                            size=self.windows_per_user,
                            replace=False
                        )
                        for idx in sorted(chosen_idx.tolist()):
                            samples.append(window_candidates[idx])
                    else:
                        samples.extend(window_candidates[:self.windows_per_user])

                # 始终保留一个“真实训练目标”样本（history -> final target）
                samples.append((user_id, history[-self.window_size:], target))
            else:
                samples.append((user_id, history[-self.window_size:], target))

        return samples

    def resample_samples(self, epoch: int):
        """
        每个 epoch 固定重采样一次窗口，保证增强可复现。
        """
        if self.use_sliding_window and self.sliding_window_mode == "sample_per_epoch":
            rng = np.random.RandomState(self.seed + epoch)
            self.samples = self._build_samples(rng=rng)
        else:
            self.samples = self._build_samples()
    
    def _item_to_tokens(self, item_id:int) -> List[int]:
        """
        item_id -> 该 item 的 SID token 列表
        code + code_offset = semantic_id(token) 
        semantic_ids[item_id] = [45, 12, 88]  （RQ-VAE 的3层code）
        加 offset 后 = [48, 15, 91]           （避免和特殊token冲突）
        """
        codes = self.semantic_ids[item_id]
        return [code2token(c) for c in codes]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx:int) -> dict:
        user_id, history, target_item = self.samples[idx]
        
        # 构造input token
        # [BOS] + [所有历史item的token]
        input_tokens = [BOS_TOKEN]
        
        # 截断历史序列：只用 max_seq_len 个 item
        # 从左边截断，保留最近的历史序列
        recent_histories = history[-self.max_seq_len:]
        
        for item_id in recent_histories:
            input_tokens.extend(self._item_to_tokens(item_id))  # [1 + len(recent_his)*3]
        
        target_tokens = self._item_to_tokens(target_item)
        
        # 左 Padding
        # 保证最后一个有效token在序列末尾，便于解码和监督对齐
        cur_len = len(input_tokens)
        pad_len = self.max_tokens - cur_len
        
        if pad_len < 0:
            # 序列过长（理论上不会，上面已经截断过了）
            input_tokens = input_tokens[-self.max_tokens:]
            pad_len = 0
        
        input_ids = [PAD_TOKEN]*pad_len + input_tokens
        attention_mask = [0]*pad_len + [1]*len(input_tokens)
        
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "input_ids": torch.LongTensor(input_ids),    # [max_tokens]
            "attention_mask": torch.LongTensor(attention_mask), # [max_tokens]
            "target_ids": torch.LongTensor(target_tokens),  # [num_rq_layers]
            "target_item": torch.tensor(target_item, dtype=torch.long)    # 标量, 评估用
        }

    @property
    def vocab_size(self) -> int:
        """
        词表大小(token的总数) = 特殊token数(BOS,PAD,EOS) + codebook_size(256)
        """
        codebook_size = self.semantic_ids.max() + 1 # 255 + 1
        return CODE_OFFSET + codebook_size
    



class SeqEvalDataset(Dataset):
    """
    评估用 Dataset (和训练 Dataset 几乎相同)
    单独抽出来的原因：
      评估时历史的来源可能不同
        val 评估：历史 = train 序列（不包含 val target）
        test 评估：历史 = train 序列（不包含 val/test target）
      
      语义更清晰，明确表达"这是评估模式"
      未来可以加入评估特有的逻辑（如排除已交互item）
    """
    def __init__(self, user_histories:Dict[int, List[int]],
                 targets:Dict[int, int],
                 semantic_ids: np.ndarray,
                 max_seq_len:int = 50,
                 num_rq_layers:int = 3):
        # 代码复用
        self._inner = SeqTrainDataset(user_histories,
                                      targets,
                                      semantic_ids,
                                      max_seq_len,
                                      num_rq_layers,
                                      use_sliding_window=False)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx:int) -> Dict:
        return self._inner[idx]
    
    

# DataLoader 工厂函数
def get_rqvae_loaders(
    embeddings: np.ndarray,
    batch_size:int = 512,
    val_ratio:float = 0.1,
    num_workers:int = 0,
    seed:int = 42
):
    """
    RQ-VAE 训练的DataLoader
    将 Embedding 随机划分成 train/val
    用 val 验证集监控重建质量，决定早停
    """
    n = len(embeddings)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    
    split = int(n * (1 - val_ratio))
    train_emb = embeddings[idx[:split]]
    val_emb = embeddings[idx[split:]]
    
    train_ds = ItemEmbeddingDataset(train_emb)
    val_ds = ItemEmbeddingDataset(val_emb)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,    # 锁页内存，加速 CPU→GPU 传输
        drop_last=True  # 丢弃最后不完整的 batch,保证每个 batch 大小一致（EMA 统计更稳定）
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"RQ-VAE DataLoader:")
    print(f"  Train: {len(train_ds):,} 样本, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds):,} 样本, {len(val_loader)} batches")
    
    return train_loader, val_loader


def get_rec_loaders(
    data:dict,
    semantic_ids:np.ndarray,
    batch_size:int = 256,
    max_seq_len: int=50,
    num_rq_layers: int = 3,
    num_workers: int = 2,
    use_sliding_window: bool = True,
    sliding_window_mode: str = "all",
    window_size: int = 20,
    min_seq_len: int = 2,
    windows_per_user: int = 2,
    seed: int = 42,
):
    """
    推荐模型的 DataLoader
    
    注意 val 和 test 的历史来源：
      都用 data["train"] 作为历史
      不能用 data["val"] 更新历史（评估协议）
      
    Returns:
        - train_loader: DataLoader
        - val_loader: DataLoader
        - test_loader: DataLoader
        - vocab_size: int
    """
    common_kwargs = dict(
        semantic_ids=semantic_ids,
        max_seq_len=max_seq_len,
        num_rq_layers=num_rq_layers
    )
    # 训练集（防泄露）：只使用 data['train'] 内部信息
    # 对每个用户，将 train 序列再切一刀：
    #   history = train_seq[:-1], target = train_seq[-1]
    # 这样不会看到 val/test 目标
    train_histories = {}
    train_targets = {}
    for user_id, seq in data['train'].items():
        if len(seq) < 2:
            continue
        train_histories[user_id] = seq[:-1]
        train_targets[user_id] = seq[-1]

    # 训练集
    train_ds = SeqTrainDataset(
        user_histories=train_histories,
        targets=train_targets,
        use_sliding_window=use_sliding_window,
        sliding_window_mode=sliding_window_mode,
        window_size=window_size, # 历史窗口大小
        min_seq_len=min_seq_len,  # 最短历史长度 -> 至少需要2个item历史才能生成样本
        windows_per_user=windows_per_user,
        seed=seed,
        **common_kwargs
    )
    # val验证集：历史=train序列，目标=val target
    val_ds = SeqEvalDataset(
        user_histories={
            u: data['train'][u] for u in data['val'] if u in data['train']
        },
        targets=data['val'],
        **common_kwargs
    )
    # test测试集：历史=train序列，目标=test target
    test_ds = SeqEvalDataset(
        user_histories={
            u: data['train'][u] for u in data['test'] if u in data['train']
        },
        targets=data['test'],
        **common_kwargs
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,    # 推荐训练不 drop_last（数据量本来就不多）
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"推荐模型 DataLoader:")
    print(f"  Train: {len(train_ds):,} 样本, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds):,} 样本,  {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds):,} 样本, {len(test_loader)} batches")
    print(f"  vocab_size: {train_ds.vocab_size}")
    
    return train_loader, val_loader, test_loader, train_ds.vocab_size


def load_data(path:Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到预处理数据: {path}\n"
            f"请先运行: python -m data.preprocess"
        )
    with open(path, 'rb') as f:
        return pickle.load(f)
    

    
    
        
            
        
    
    
