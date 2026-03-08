from Amazon_Dataset import get_rec_loaders
import argparse
import pickle
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import hr_at_k, ndcg_at_k
from typing import Dict, List, Tuple
import numpy as np
from models.Transformer import CausalTransformer


def build_sid_to_item(semantic_ids:np.ndarray) -> Dict[tuple, List[int]]:
    """
    构建 Lookup Table {(c0, c1, c2): [item_id, ...]}
    为什么不是List而是单个int
        两个不同的item可能拥有相同的语义ID (发生碰撞)
        虽然概率极低(256^3 >> num_items)，但需要处理
    """
    sid2item = {}
    for item_id, sid in enumerate(semantic_ids):
        sid = tuple(sid)
        if sid not in sid2item:
            sid2item[sid] = []
        sid2item[sid].append(item_id)
    return sid2item


def calculate_metrics(
    recommended: List[int],
    target:int,
    topk:List[int]
)->Dict[str, float]:
    """
    对单个用户计算所有K值的指标：
    Returns: {"HR@1": ..., "NDCG@1": ..., "HR@5": ..., ...}
    """
    metrics = {}
    for k in topk:
        metrics[f"HR@{k}"] = hr_at_k(recommended, target, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(recommended, target, k)
    return metrics


def print_metrics(
    metrics:Dict[str, float],
    topk: List[int],
    prefix: str = ""
):
    """
    格式化打印评估指标
    """
    parts = []
    for k in topk:
        parts.append(
            f"HR@{k}={metrics[f'HR@{k}']:.4f} "
            f"NDCG@{k}={metrics[f'NDCG@{k}']:.4f}"
        )
    line = " | ".join(parts)
    print(f'{prefix} | {line}'if prefix else line)

    

def beam_to_candidate(
    beams: torch.tensor,    # [batch_size, beam_size, L]
    sid2item: Dict[tuple, List[int]],
    code_offset:int = 3
)-> List[List[int]]:
    """
    把 Beam Search 输出结果转换成推荐 item 列表
    Input: 
        - beams: [batch_size, beam_size, num_rq_layers], token ID(含code_offset)
    Output:
        - List[List[int]], 外层->用户，内层->推荐列表（按beam search结果排序）
    """
    num_user, beam_size, L = beams.shape
    all_candidates = []
    
    for u in range(num_user):
        candidates = []
        seen = set()
        
        for beam_idx in range(beam_size):
            # token ID -> raw code
            raw_codes = tuple(
                beams[u, beam_idx, l].item() - code_offset for l in range(L)
            )
            # 查表
            items = sid2item.get(raw_codes, [])
            for item in items:
                if item not in seen:
                    candidates.append(item)
                    seen.add(item)
        all_candidates.append(candidates)
    
    return all_candidates


def build_prefix_to_next_tokens(
    sid2item: Dict[tuple, List[int]],
    code_offset: int = 3
) -> Dict[tuple, List[int]]:
    """
    根据有效 semantic id 构建前缀约束表：
    {prefix_tokens: [allowed_next_token, ...]}
    """
    prefix_to_next = {}
    for raw_sid in sid2item.keys():
        for depth in range(len(raw_sid)):
            prefix = tuple(raw_sid[i] + code_offset for i in range(depth))
            next_token = raw_sid[depth] + code_offset
            if prefix not in prefix_to_next:
                prefix_to_next[prefix] = set()
            prefix_to_next[prefix].add(next_token)

    return {k: sorted(v) for k, v in prefix_to_next.items()}


@torch.inference_mode()
def generate_beam_constrained(
    model: CausalTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    user_ids: torch.Tensor,
    beam_size: int,
    prefix_to_next: Dict[tuple, List[int]],
) -> torch.Tensor:
    """
    基于有效 SID 前缀的约束 Beam Search。
    仅允许扩展到数据中存在的 semantic id 前缀，避免候选 tuple 大量无效。
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device
    L = model.num_rq_layers
    V = model.vocab_size

    beams = torch.zeros((B, 1, L), dtype=torch.long, device=device)
    beam_scores = torch.zeros((B, 1), dtype=torch.float32, device=device)

    for step in range(L):
        curr_beam = beams.size(1)

        exp_input = input_ids.unsqueeze(1).expand(-1, curr_beam, -1).reshape(B * curr_beam, T)
        exp_mask = attention_mask.unsqueeze(1).expand(-1, curr_beam, -1).reshape(B * curr_beam, T)

        if step > 0:
            prefixes_flat = beams[:, :, :step].reshape(B * curr_beam, step)
            ext_ids = torch.cat([exp_input, prefixes_flat], dim=1)
            ext_mask = torch.cat([
                exp_mask,
                torch.ones(B * curr_beam, step, dtype=attention_mask.dtype, device=device)
            ], dim=1)
        else:
            ext_ids = exp_input
            ext_mask = exp_mask

        exp_user_ids = user_ids.unsqueeze(1).expand(-1, curr_beam).reshape(B * curr_beam)
        logits = model(ext_ids, ext_mask, exp_user_ids)[:, -1, :]
        log_prob = torch.log_softmax(logits, dim=-1)

        masked_log_prob = torch.full_like(log_prob, float('-inf'))

        if step > 0:
            prefixes = prefixes_flat.tolist()
        else:
            prefixes = [()] * (B * curr_beam)

        for i, prefix in enumerate(prefixes):
            key = tuple(prefix)
            allowed = prefix_to_next.get(key, [])
            if allowed:
                allowed_idx = torch.tensor(allowed, dtype=torch.long, device=device)
                masked_log_prob[i, allowed_idx] = log_prob[i, allowed_idx]

        masked_log_prob = masked_log_prob.view(B, curr_beam, V)
        total_scores = beam_scores.unsqueeze(-1) + masked_log_prob
        flat_scores = total_scores.view(B, curr_beam * V)

        k = min(beam_size, flat_scores.size(1))
        new_scores, new_pos = torch.topk(flat_scores, k, dim=-1)
        parent_idx = new_pos // V
        token_idx = new_pos % V

        new_beams = beams.gather(1, parent_idx.unsqueeze(-1).expand(-1, -1, L))
        new_beams[:, :, step] = token_idx

        beams = new_beams
        beam_scores = new_scores

    return beams


@torch.inference_mode()
def evaluate(
    model: CausalTransformer,
    loader: DataLoader,
    sid2item: Dict[tuple, List[int]],
    topk: List[int],
    beam_size:int,
    device:str,
    split:str = 'val',
    print_hit_samples: int = 0
) -> Dict[str, float]:
    """
    在给定DataLoader上完整评估
    流程：
        1. Beam Search 生成候选语义ID
        2. 查表转换成候选 item 列表
        3. 逐用户计算 HR@K 和 NDCG@K
        4. 返回所有指标的均值
    params:
        - split:日志前缀，'val'或'test'
    Returns:
        {"HR@1": 0.032, "NDCG@1": 0.032,
       "HR@5": 0.098, "NDCG@5": 0.071, ...}
    """
    
    print(f"Using {device}.. decode_mode=constrained")

    code_offset = model.CODE_OFFSET
    prefix_to_next = build_prefix_to_next_tokens(sid2item, code_offset=code_offset)
    
    model.eval()
    
    # 初始化累计指标
    total_metrics = {f"HR@{k}":0.0 for k in topk}
    total_metrics.update({f"NDCG@{k}":0.0 for k in topk})
    total_num_users = 0
    total_empty_candidates = 0
    total_candidate_count = 0
    hit_samples_printed = 0
    
    for batch in tqdm(loader, desc=f"[{split}]", leave=False):
        user_ids = batch['user_id'].to(device)  # [B]
        input_ids = batch['input_ids'].to(device)   # [B, T]
        attention_mask = batch['attention_mask'].to(device) # [B, T]
        target_items = batch['target_item'].squeeze(-1).tolist()   # List[int]
        
        num_user = input_ids.size(0)
        
        # Beam Search: 生成候选语义ID
        beams = generate_beam_constrained(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            beam_size=beam_size,
            prefix_to_next=prefix_to_next
        )
        
        # 转换成推荐列表
        all_candidates = beam_to_candidate(beams, sid2item, code_offset=code_offset)
        total_empty_candidates += sum(1 for candidates in all_candidates if len(candidates) == 0)
        total_candidate_count += sum(len(candidates) for candidates in all_candidates)
        
        # 逐用户计算指标并累计
        for u in range(num_user):
            user_metrics = calculate_metrics(
                recommended=all_candidates[u],
                target=target_items[u],
                topk=topk
                )
            for key, val in user_metrics.items():
                total_metrics[key] += val

            if print_hit_samples > 0 and hit_samples_printed < print_hit_samples:
                target = target_items[u]
                if target in all_candidates[u]:
                    rank = all_candidates[u].index(target) + 1
                    matched_beam = None
                    matched_codes = None
                    matched_items = None

                    for beam_idx in range(beams.size(1)):
                        codes = tuple(
                            beams[u, beam_idx, l].item() - code_offset
                            for l in range(beams.size(2))
                        )
                        items = sid2item.get(codes, [])
                        if target in items:
                            matched_beam = beam_idx
                            matched_codes = codes
                            matched_items = items
                            break

                    valid_tokens = input_ids[u][attention_mask[u].bool()].tolist()
                    print(f"[{split}] HIT sample#{hit_samples_printed + 1}: user_in_batch={u}, target_item={target}, rank={rank}")
                    print(f"  history_tokens_tail={valid_tokens[-12:]}")
                    if matched_beam is not None:
                        sid_tokens = beams[u, matched_beam].tolist()
                        print(f"  matched_beam={matched_beam}, sid_tokens={sid_tokens}, sid_codes={matched_codes}, items={matched_items[:5]}")
                    else:
                        print("  matched_beam=not_found_in_top_beams (target may come from dedup/collision expansion)")

                    hit_samples_printed += 1
        total_num_users += num_user
    
    # 取均值
    for key in total_metrics:
        total_metrics[key] /= total_num_users

    empty_ratio = total_empty_candidates / max(total_num_users, 1)
    avg_candidate_num = total_candidate_count / max(total_num_users, 1)
    print(f"[{split}] empty_candidate_ratio={empty_ratio:.4f}, avg_candidate_num={avg_candidate_num:.2f}")
    
    return total_metrics


# 加载 checkpoint 评估
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--print_hit_samples', type=int, default=0,
                        help='打印命中样本的详细信息，0表示不打印')
    args = parser.parse_args()
    
    # 加载checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"From Epoch {ckpt['epoch']}")
    
    # Loading Data
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    semantic_ids = np.load(config['sid_path'])
    if semantic_ids.ndim != 2:
        raise ValueError(f"semantic_ids must be 2D [num_items, num_rq_layers], got shape={semantic_ids.shape}")

    sid_num_rq_layers = int(semantic_ids.shape[1])
    cfg_num_rq_layers = int(config.get('num_rq_layers', sid_num_rq_layers))
    if cfg_num_rq_layers != sid_num_rq_layers:
        print(
            f"[WARN] num_rq_layers from checkpoint config={cfg_num_rq_layers} "
            f"!= sid layers={sid_num_rq_layers}, using sid layers."
        )
    num_rq_layers = sid_num_rq_layers

    target_loss_weights = config.get('target_loss_weights')
    if target_loss_weights is None or len(target_loss_weights) != num_rq_layers:
        target_loss_weights = [1.0] * num_rq_layers

    sid2item = build_sid_to_item(semantic_ids)
    
    # Build DataLoader
    eval_batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    eval_num_workers = args.num_workers if args.num_workers is not None else config['num_workers']

    _, val_loader, test_loader, vocab_size = get_rec_loaders(
        data=data,
        semantic_ids=semantic_ids,
        batch_size=eval_batch_size,
        max_seq_len=config['max_seq_len'],
        num_rq_layers=num_rq_layers,
        num_workers=eval_num_workers
    )
    loader = test_loader if args.split == 'test' else val_loader
    
    # initialize model
    max_tokens = 1 + config['max_seq_len'] * num_rq_layers
    all_user_ids = set(data['train'].keys()) | set(data['val'].keys()) | set(data['test'].keys())
    num_users = max(all_user_ids) + 1

    model = CausalTransformer(
        vocab_size=vocab_size,
        num_users=num_users,
        d_model=config['d_model'],
        num_head=config['num_head'],
        num_layers=config['num_layers'],
        dim_ffn=config['dim_feedforward'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=max_tokens + num_rq_layers,
        num_rq_layers=num_rq_layers,
        codebook_size=config['codebook_size'],
        use_user_token=config.get('use_user_token', True),
        target_loss_weights=target_loss_weights
    ).to(device)
    
    model.load_state_dict(ckpt['model_state'])
    
    metrics = evaluate(
        model=model,
        loader=loader,
        sid2item=sid2item,
        topk=config['topk'],
        beam_size=args.beam_size,
        device=device,
        split=args.split.capitalize(),
        print_hit_samples=args.print_hit_samples
    )
    
    print(f"\n{args.split.capitalize()} 结果：")
    print_metrics(metrics, config['topk'])
    

if __name__ == "__main__":
    main()