from Amazon_Dataset import get_rec_loaders
import argparse
import pickle
import time
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from metrics import hr_at_k, ndcg_at_k
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from models.Transformer import CausalTransformer


def build_sid_to_item_tables(
    semantic_ids:np.ndarray
) -> tuple[Dict[tuple, int], Dict[tuple, List[int]]]:
    """
    将 SID -> item 拆成：
    - sid2item_single: 大多数 1对1 SID，直接映射到单个 item_id
    - sid2item_multi: 少数碰撞 SID, 映射到多个 item_id
    """
    sid_buckets = defaultdict(list)
    for item_id, sid in enumerate(semantic_ids):
        sid_buckets[tuple(map(int, sid))].append(item_id)
        
    sid2item_single = {}
    sid2item_multi = {}
    for sid, items in sid_buckets.items():
        if len(items) == 1:
            sid2item_single[sid] = items[0]
        else:
            sid2item_multi[sid] = items
    
    return sid2item_single, sid2item_multi


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


def parse_int_list(arg: str | None) -> list[int] | None:
    if arg is None:
        return None
    values = [int(x.strip()) for x in arg.split(",") if x.strip()]
    return values if values else None




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
    beams: torch.Tensor,    # [batch_size, beam_size, L]
    sid2item_single: Dict[tuple, int],
    sid2item_multi: Dict[tuple, List[int]],
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

    # 避免在 Python 循环里频繁 .item() 触发 GPU 同步。
    raw_codes = beams.detach().to("cpu", copy=False).numpy() - code_offset
    flat_codes = raw_codes.reshape(num_user * beam_size, L)

    # 批内相同 SID 只查一次表，减少重复 dict lookup。
    sid_lookup_cache = {}
    flat_items = []
    
    for row in flat_codes.tolist():
        sid = tuple(row)
        if sid not in sid_lookup_cache:
            single_item = sid2item_single.get(sid)
            if single_item is not None:
                sid_lookup_cache[sid] = single_item
            else:
                sid_lookup_cache[sid] = sid2item_multi.get(sid, [])
        flat_items.append(sid_lookup_cache[sid])

    all_candidates = []
    for user_idx in range(num_user):
        candidates = []
        seen = set()
        start = user_idx * beam_size
        end = start + beam_size

        for items in flat_items[start:end]:
            if isinstance(items, int):
                if items not in seen:
                    candidates.append(items)
                    seen.add(items)
            else:
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


def build_prefix_branch_tables(
    sid2item: Dict[tuple, List[int]],
    code_offset: int = 3
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    构建逐层前缀约束表：
    - allowed_masks[step][state, token] -> 当前前缀状态下第 branch 个可选 token
    - next_state_tables[step][state, token] -> 选择 token 后进入的下一层前缀状态, 最后一步填-1
    - branch_masks[step][state, branch] -> 该 branch 是否有效
    """
    raw_sids = list(sid2item.keys())
    if not raw_sids:
        raise ValueError("sid2item is empty; cannot build constrained beam tables")

    num_rq_layers = len(raw_sids[0])
    prefix_children_by_depth = [defaultdict(set) for _ in range(num_rq_layers)]

    for raw_sid in raw_sids:
        token_sid = tuple(code + code_offset for code in raw_sid)
        for depth in range(num_rq_layers):
            prefix = token_sid[:depth]
            prefix_children_by_depth[depth][prefix].add(token_sid[depth])

    prefix_state_maps = []
    for depth in range(num_rq_layers):
        prefixes = sorted(prefix_children_by_depth[depth].keys())
        prefix_state_maps.append({prefix: idx for idx, prefix in enumerate(prefixes)})

    allowed_masks = []
    next_state_tables = []
    branch_masks = []

    for depth in range(num_rq_layers):
        state_map = prefix_state_maps[depth]
        next_state_map = prefix_state_maps[depth + 1] if depth < num_rq_layers - 1 else None
        
        max_branch = max(len(children) for children in prefix_children_by_depth[depth].values())
        token_table = torch.full((len(state_map), max_branch), -1, dtype=torch.long)
        next_state_table = torch.full((len(state_map), max_branch), -1, dtype=torch.long)
        branch_mask = torch.zeros((len(state_map), max_branch), dtype=torch.bool)

        for prefix, row_idx in state_map.items():
            children = sorted(prefix_children_by_depth[depth][prefix])
            for branch_idx, token in enumerate(children):
                token_table[row_idx, branch_idx] = token
                branch_mask[row_idx, branch_idx] = True
                if next_state_map is not None:
                    next_state_table[row_idx, branch_idx] = next_state_map[prefix + (token,)]
                
            
        allowed_masks.append(token_table)
        next_state_tables.append(next_state_table)
        branch_masks.append(branch_mask)

    return allowed_masks, next_state_tables, branch_masks



def move_branch_tables_to_device(
    allowed_tokens: list[torch.Tensor],
    next_state_tables: list[torch.Tensor],
    branch_masks:list[torch.Tensor],
    device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    return (
        [table.to(device, non_blocking=True) for table in allowed_tokens],
        [table.to(device, non_blocking=True) for table in next_state_tables],
        [mask.to(device, non_blocking=True) for mask in branch_masks]
    )


def build_dynamic_beam_schedule(max_beam_size: int, num_steps: int) -> List[int]:
    """
    默认动态 beam schedule：
    - L=4, beam=20 -> [20, 20, 20, 20]
    - L=4, beam=10 -> [10, 10, 10, 10]
    """
    if num_steps <= 0:
        return []
    return [max(1, max_beam_size)] * num_steps


def normalize_beam_schedule(
    beam_size: int,
    num_steps: int,
    beam_schedule: List[int] | None = None
) -> List[int]:
    if beam_schedule is None:
        return build_dynamic_beam_schedule(beam_size, num_steps)

    if len(beam_schedule) != num_steps:
        raise ValueError(
            f"beam_schedule length ({len(beam_schedule)}) must equal num_rq_layers ({num_steps})"
        )

    normalized = [max(1, min(beam_size, int(v))) for v in beam_schedule]
    for idx in range(1, len(normalized)):
        normalized[idx] = min(normalized[idx - 1], normalized[idx])
    return normalized



@torch.inference_mode()
def generate_beam_constrained(
    model: CausalTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    user_ids: torch.Tensor,
    beam_size: int,
    allowed_tokens: list[torch.Tensor],
    next_states: list[torch.Tensor],
    branch_masks:list[torch.Tensor],
    beam_schedule: List[int] | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> torch.tensor:
    """
    基于有效 SID 前缀的约束 Beam Search。
    仅允许扩展到数据中存在的 semantic id 前缀，避免候选 tuple 大量无效。
    """
    model.eval()
    B, T = input_ids.shape
    device = input_ids.device
    L = model.num_rq_layers
    V = model.vocab_size
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    resolved_beam_schedule = normalize_beam_schedule(beam_size, L, beam_schedule)
    compact_input_ids, compact_attention_mask, _ = model.prepare_compact_inputs(
        input_ids, attention_mask
    )
    compact_width = compact_input_ids.size(1)

    beams = torch.zeros((B, 1, L), dtype=torch.long, device=device)
    beam_scores = torch.zeros((B, 1), dtype=torch.float32, device=device)
    beam_states = torch.zeros((B, 1), dtype=torch.long, device=device)

    for step in range(L):
        curr_beam = beams.size(1)
        exp_compact_ids = compact_input_ids.unsqueeze(1).expand(-1, curr_beam, -1).reshape(B * curr_beam, compact_width)
        exp_compact_mask = compact_attention_mask.unsqueeze(1).expand(-1, curr_beam, -1).reshape(B * curr_beam, compact_width)
        prefixes_flat = beams[:, :, :step].reshape(B * curr_beam, step) if step > 0 else None

        exp_user_ids = user_ids.unsqueeze(1).expand(-1, curr_beam).reshape(B * curr_beam)
        with torch.autocast(
            device_type=autocast_device_type,
            dtype=amp_dtype,
            enabled=amp_enabled
        ):
            logits = model.decode_last_logits(
                exp_compact_ids,
                exp_compact_mask,
                exp_user_ids,
                prefix_ids=prefixes_flat
            )

        state_ids = beam_states.reshape(-1)

        step_tokens = allowed_tokens[step].index_select(0, state_ids)          # [B*beam, max_branch]
        step_branch_mask = branch_masks[step].index_select(0, state_ids)       # [B*beam, max_branch]
        step_tokens_safe = step_tokens.clamp_min(0)

        # 基于 full-vocab 的 logsumexp 做归一化
        logits_f = logits.float()
        log_norm = torch.logsumexp(logits_f, dim=-1, keepdim=True)             # [B*beam, 1]
        branch_logits = logits_f.gather(1, step_tokens_safe)                   # [B*beam, max_branch]
        branch_log_prob = branch_logits - log_norm
        branch_log_prob = branch_log_prob.masked_fill(~step_branch_mask, float("-inf"))

        max_branch = step_tokens.size(1)
        branch_log_prob = branch_log_prob.view(B, curr_beam, max_branch)
        step_tokens = step_tokens.view(B, curr_beam, max_branch)

        total_scores = beam_scores.unsqueeze(-1) + branch_log_prob
        flat_scores = total_scores.view(B, curr_beam * max_branch)

        step_beam_size = resolved_beam_schedule[step]
        k = min(step_beam_size, flat_scores.size(1))
        new_scores, new_pos = torch.topk(flat_scores, k, dim=-1)

        parent_idx = new_pos // max_branch
        branch_idx = new_pos % max_branch

        new_beams = beams.gather(1, parent_idx.unsqueeze(-1).expand(-1, -1, L))
        selected_tokens = step_tokens.gather(1, parent_idx.unsqueeze(-1).expand(-1, -1, max_branch))
        token_idx = selected_tokens.gather(2, branch_idx.unsqueeze(-1)).squeeze(-1)
        new_beams[:, :, step] = token_idx

        beams = new_beams
        beam_scores = new_scores

        if step < L - 1:
            step_next_states = next_states[step].index_select(0, state_ids).view(B, curr_beam, max_branch)
            selected_next_states = step_next_states.gather(
                1, parent_idx.unsqueeze(-1).expand(-1, -1, max_branch)
            )
            beam_states = selected_next_states.gather(
                2, branch_idx.unsqueeze(-1)
            ).squeeze(-1)


    return beams

def resolve_eval_amp_settings(device: str) -> tuple[bool, torch.dtype | None]:
    if device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return False, None


@torch.inference_mode()
def evaluate(
    model: CausalTransformer,
    loader: DataLoader,
    sid2item:Dict[tuple, List[int]],
    sid2item_single: Dict[tuple, int],
    sid2item_multi: Dict[tuple, List[int]],
    topk: List[int],
    beam_size:int,
    device:str,
    beam_schedule: List[int] | None = None,
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
    
    if max(topk) > beam_size:
        raise ValueError(
            f"max(topk)={max(topk)} exceeds beam_size={beam_size}. "
            "Increase beam_size or reduce topk to avoid truncated metrics."
        )

    amp_enabled, amp_dtype = resolve_eval_amp_settings(device)
    amp_dtype_name = "fp32" if amp_dtype is None else "bf16"
    resolved_beam_schedule = normalize_beam_schedule(beam_size, model.num_rq_layers, beam_schedule)
    print(
        f"Using {device}.. decode_mode=constrained "
        f"beam_schedule={resolved_beam_schedule} eval_amp={amp_dtype_name}"
    )

    code_offset = model.CODE_OFFSET
    allowed_tokens, next_states, branch_masks = build_prefix_branch_tables(
        sid2item,
        code_offset=code_offset
    )
    allowed_tokens, next_states, branch_masks = move_branch_tables_to_device(
        allowed_tokens,
        next_states,
        branch_masks,
        torch.device(device)
    )

    
    model.eval()
    
    # 初始化累计指标
    total_metrics = {f"HR@{k}":0.0 for k in topk}
    total_metrics.update({f"NDCG@{k}":0.0 for k in topk})
    total_num_users = 0
    total_empty_candidates = 0
    total_candidate_count = 0
    hit_samples_printed = 0
    total_decode_time = 0.0
    total_candidate_time = 0.0
    total_metric_time = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc=f"[{split}]", leave=False):
        user_ids = batch['user_id'].to(device, non_blocking=True)  # [B]
        input_ids = batch['input_ids'].to(device, non_blocking=True)   # [B, T]
        attention_mask = batch['attention_mask'].to(device, non_blocking=True) # [B, T]
        target_items = batch['target_item'].squeeze(-1).tolist()   # List[int]
        
        num_user = input_ids.size(0)
        
        # Beam Search: 生成候选语义ID
        decode_start = time.perf_counter()
        beams = generate_beam_constrained(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_ids=user_ids,
            beam_size=beam_size,
            allowed_tokens=allowed_tokens,
            next_states=next_states,
            branch_masks=branch_masks,
            beam_schedule=resolved_beam_schedule,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        total_decode_time += time.perf_counter() - decode_start
        
        # 转换成推荐列表
        candidate_start = time.perf_counter()
        
        all_candidates = beam_to_candidate(
            beams, 
            sid2item_single,
            sid2item_multi,  
            code_offset=code_offset
        )
        
        total_empty_candidates += sum(1 for candidates in all_candidates if len(candidates) == 0)
        total_candidate_count += sum(len(candidates) for candidates in all_candidates)
        total_candidate_time += time.perf_counter() - candidate_start
        
        # 逐用户计算指标并累计
        metric_start = time.perf_counter()
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
        total_metric_time += time.perf_counter() - metric_start
        num_batches += 1
    
    # 取均值
    for key in total_metrics:
        total_metrics[key] /= total_num_users

    empty_ratio = total_empty_candidates / max(total_num_users, 1)
    avg_candidate_num = total_candidate_count / max(total_num_users, 1)
    avg_decode_time_per_batch = total_decode_time / max(num_batches, 1)
    avg_decode_time_per_user = total_decode_time / max(total_num_users, 1)
    print(f"[{split}] empty_candidate_ratio={empty_ratio:.4f}, avg_candidate_num={avg_candidate_num:.2f}")
    print(
        f"[{split}] timing decode={total_decode_time:.2f}s "
        f"candidate={total_candidate_time:.2f}s metric={total_metric_time:.2f}s "
        f"avg_decode_per_batch={avg_decode_time_per_batch:.3f}s "
        f"avg_decode_per_user={avg_decode_time_per_user:.6f}s"
    )
    
    return total_metrics


# 加载 checkpoint 评估
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--beam_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--print_hit_samples', type=int, default=0,
                        help='打印命中样本的详细信息，0表示不打印')
    
    parser.add_argument('--topk', type=str, default=None,
                    help='comma-separated topk, e.g. 1,5,10')
    parser.add_argument('--beam_schedule', type=str, default=None,
                        help='comma-separated beam schedule, e.g. 10,10,10,10')
    
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
    sid2item_single, sid2item_multi = build_sid_to_item_tables(semantic_ids)
    
    # Build DataLoader
    eval_batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    eval_num_workers = args.num_workers if args.num_workers is not None else config['num_workers']
    eval_beam_size = args.beam_size if args.beam_size is not None else config.get('beam_size', 20)

    eval_topk = parse_int_list(args.topk) if args.topk is not None else config['topk']
    eval_beam_schedule = (
        parse_int_list(args.beam_schedule)
        if args.beam_schedule is not None
        else (
            config.get('beam_schedule')
            if eval_beam_size == config.get('beam_size', eval_beam_size)
            else None
        )
    )


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
        target_loss_weights=target_loss_weights,
        hierarchical_attention_enabled=config.get('hierarchical_attention_enabled', False),
        attention_layout=config.get('attention_layout')
    ).to(device)
    
    model.load_state_dict(ckpt['model_state'])
    
    metrics = evaluate(
        model=model,
        loader=loader,
        sid2item=sid2item,
        sid2item_single=sid2item_single,
        sid2item_multi=sid2item_multi,
        topk=eval_topk,
        beam_size=eval_beam_size,
        device=device,
        beam_schedule=eval_beam_schedule,
        split=args.split.capitalize(),
        print_hit_samples=args.print_hit_samples
    )

    print(f"\n{args.split.capitalize()} 结果：")
    print_metrics(metrics, eval_topk)

    

if __name__ == "__main__":
    main()
