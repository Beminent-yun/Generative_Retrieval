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
    """
    def __init__(self, user_histories: Dict[int, List[int]],    # {user_id:[item_id,...]}
                 targets: Dict[int, int],   # {user_id:target_item_id}
                 semantic_ids: np.ndarray,  # Look up table (num_items, num_rq_layers)
                 max_seq_len:int = 50,  # 用户历史行为序列的最大长度（item数）
                 num_rq_layers:int = 3):    # 每个item对应几个token/code
        if semantic_ids.ndim != 2:
            raise ValueError(f"semantic_ids must be 2D [num_items, num_rq_layers], got shape={semantic_ids.shape}")

        sid_layers = semantic_ids.shape[1]
        if num_rq_layers != sid_layers:
            raise ValueError(
                f"num_rq_layers ({num_rq_layers}) does not match semantic_ids second dim ({sid_layers})"
            )

        self.semantic_ids = semantic_ids
        self.max_seq_len = max_seq_len
        self.num_rq_layers = num_rq_layers
        
        # max_tokens: BOS + max_seq_len * num_rq_layers(1id->num_rq_layers token)
        self.max_tokens = 1 + max_seq_len * num_rq_layers
        
        # 构建样本列表
        self.samples = []
        for user_id, history in user_histories.items():
            if user_id not in targets:
                continue
            if len(history) == 0:
                continue
            self.samples.append((int(user_id), history, targets[user_id]))

        print(f"SeqTrainDataset: {len(self.samples):,} 个样本")
    
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
                                      num_rq_layers)

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
    num_workers: int = 2
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
    

    
    
        
            
        
    
    