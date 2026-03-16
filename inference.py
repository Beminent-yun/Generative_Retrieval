import argparse
import pickle
import numpy as np
import torch
from Amazon_Dataset import BOS_TOKEN, PAD_TOKEN, code2token
from evaluate import (
    beam_to_candidate,
    build_prefix_branch_tables,
    build_sid_to_item,
    generate_beam_constrained,
    move_branch_tables_to_device
)
from models.Transformer import CausalTransformer


def build_model_input(
    history_items: list[int],
    semantic_ids: np.ndarray,
    max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For user:
        history item_id 列表 -> left-padded input_ids / attention_mask
    """
    recent_history = history_items[-max_seq_len:]
    
    # item -> token
    input_tokens = [BOS_TOKEN]
    for item_id in recent_history:
        codes = semantic_ids[item_id]
        input_tokens.extend(code2token(int(c)) for c in codes)
    
    # left padding & generate attention_mask
    max_tokens = 1 + max_seq_len * semantic_ids.shape[1]
    pad_len = max_tokens - len(input_tokens)
    if pad_len < 0:
        raise ValueError(
            f"input is longer than max_tokens = {max_tokens}, "
            f"got len(input_tokens)={len(input_tokens)}"
        )
    input_ids = [PAD_TOKEN] * pad_len + input_tokens
    attention_mask = [0]*pad_len + [1]*len(input_tokens)
    
    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([attention_mask], dtype=torch.long)
    )


def load_model_and_tables(checkpoint_path: str, device:str):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    semantic_ids = np.load(config['sid_path'])
    
    sid_num_rq_layers = int(semantic_ids.shape[1])
    max_tokens = 1 + config['max_seq_len'] * sid_num_rq_layers
    
    all_user_ids = set(data['train'].keys()) | set(data['val'].keys()) | set(data['test'].keys())
    num_users = max(all_user_ids) + 1
    vocab_size = int(semantic_ids.max()) + 1 + 3
    
    target_loss_weights = config.get('target_loss_weights')
    if target_loss_weights is None or len(target_loss_weights) != sid_num_rq_layers:
        target_loss_weights = [1.0] * sid_num_rq_layers
    
    model = CausalTransformer(
        vocab_size=vocab_size,
        num_users=num_users,
        d_model=config["d_model"],
        num_head=config["num_head"],
        num_layers=config["num_layers"],
        dim_ffn=config["dim_feedforward"],
        dropout_rate=config["dropout_rate"],
        max_seq_len=max_tokens + sid_num_rq_layers,
        num_rq_layers=sid_num_rq_layers,
        codebook_size=config["codebook_size"],
        use_user_token=config.get("use_user_token", True),
        target_loss_weights=target_loss_weights,
    ).to(device)
    
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    sid2item = build_sid_to_item(semantic_ids)
    allowed_tokens, next_states, branch_masks = build_prefix_branch_tables(
        sid2item,
        vocab_size=model.vocab_size,
        code_offset=model.CODE_OFFSET
    )
    allowed_tokens, next_states, branch_masks = move_branch_tables_to_device(
        allowed_tokens,
        next_states,
        branch_masks,
        torch.device(device)
    )
    
    return model, config, semantic_ids, sid2item, allowed_tokens, next_states, branch_masks


@torch.inference_mode
def recommend_next_items(
    model: CausalTransformer,
    history_items: list[int],
    semantic_ids: np.ndarray,
    sid2item,
    allowed_tokens,
    next_states,
    branch_masks,
    max_seq_len:int,
    beam_size:int,
    device:str,
    user_id:int = 0
)->tuple[list[int], torch.Tensor]:
    
    input_ids, attention_mask = build_model_input(
        history_items=history_items,
        semantic_ids=semantic_ids,
        max_seq_len=max_seq_len
    )
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    user_ids = torch.tensor([user_id], dtype=torch.long, device=device)
    
    beams = generate_beam_constrained(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        user_ids=user_ids,
        beam_size=beam_size,
        allowed_tokens=allowed_tokens,
        next_states=next_states,
        branch_masks=branch_masks,
        beam_schedule=None,
        amp_enabled=(device == 'cuda' and torch.cuda.is_bf16_supported()),
        amp_dtype=(torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else None)
    )
    
    candidates = beam_to_candidate(
        beams=beams,
        sid2item=sid2item,
        code_offset=model.CODE_OFFSET
    )[0]
    
    return candidates, beams[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--history', type=str, required=True, help='comma-separated item ids, e.g. 12, 45, 90')
    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--user_id', type=int, default=0)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model, config, semantic_ids, sid2item, allowed_tokens, next_states, branch_masks = load_model_and_tables(
        args.checkpoint,
        device
    )
    
    history_items = [int(x) for x in args.history.split(',') if x.strip()]
    candidates, beams = recommend_next_items(
        model=model,
        history_items=history_items,
        semantic_ids=semantic_ids,
        sid2item=sid2item,
        allowed_tokens=allowed_tokens,
        next_states=next_states,
        branch_masks=branch_masks,
        max_seq_len=config['max_seq_len'],
        beam_size=args.beam_size,
        device=device,
        user_id=args.user_id
    )
    
    print(f"History: {history_items}")
    print(f"Top-{args.topk} recommended items: {candidates[:args.topk]}")
    print(f"Top beams (token ids):\n{beams}")
    

if __name__ == "__main__":
    main()