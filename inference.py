import argparse
import pickle
from typing import Any

import numpy as np
import torch

from Amazon_Dataset import BOS_TOKEN, PAD_TOKEN, code2token
from evaluate import (
    beam_to_candidate,
    build_prefix_branch_tables,
    build_sid_to_item,
    build_sid_to_item_tables,
    generate_beam_constrained,
    move_branch_tables_to_device,
)
from models.Transformer import CausalTransformer


def build_model_input(
    history_items: list[int],
    semantic_ids: np.ndarray,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    history item_id 列表 -> left-padded input_ids / attention_mask
    """
    recent_history = history_items[-max_seq_len:]

    input_tokens = [BOS_TOKEN]
    for item_id in recent_history:
        codes = semantic_ids[item_id]
        input_tokens.extend(code2token(int(c)) for c in codes)

    max_tokens = 1 + max_seq_len * semantic_ids.shape[1]
    pad_len = max_tokens - len(input_tokens)
    if pad_len < 0:
        raise ValueError(
            f"input is longer than max_tokens={max_tokens}, "
            f"got len(input_tokens)={len(input_tokens)}"
        )

    input_ids = [PAD_TOKEN] * pad_len + input_tokens
    attention_mask = [0] * pad_len + [1] * len(input_tokens)

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([attention_mask], dtype=torch.long),
    )


def get_item_title(data: dict[str, Any], item_id: int) -> str:
    titles = data.get("item_titles")
    if titles is not None and 0 <= item_id < len(titles):
        title = titles[item_id]
        if isinstance(title, str) and title.strip():
            return title.strip()
    return f"item_id={item_id}"


def get_user_history_for_inference(
    data: dict[str, Any],
    user_id: int,
    split: str,
) -> tuple[list[int], int | None]:
    train = data["train"]
    val = data["val"]
    test = data["test"]

    if split == "train":
        if user_id not in train:
            raise ValueError(f"user_id={user_id} not found in train split")
        return list(train[user_id]), None

    if split == "val":
        if user_id not in train or user_id not in val:
            raise ValueError(f"user_id={user_id} must exist in both train and val for split='val'")
        return list(train[user_id]), int(val[user_id])

    if split == "test":
        if user_id not in train or user_id not in val or user_id not in test:
            raise ValueError(
                f"user_id={user_id} must exist in train, val, and test for split='test'"
            )
        history = list(train[user_id]) + [int(val[user_id])]
        return history, int(test[user_id])

    raise ValueError(f"Unsupported split: {split}")


def load_model_and_tables(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)
    semantic_ids = np.load(config["sid_path"])

    sid_num_rq_layers = int(semantic_ids.shape[1])
    max_tokens = 1 + config["max_seq_len"] * sid_num_rq_layers

    all_user_ids = set(data["train"].keys()) | set(data["val"].keys()) | set(data["test"].keys())
    num_users = max(all_user_ids) + 1
    vocab_size = int(semantic_ids.max()) + 1 + 3

    target_loss_weights = config.get("target_loss_weights")
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
        hierarchical_attention_enabled=config.get("hierarchical_attention_enabled", False),
        attention_layout=config.get("attention_layout"),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sid2item = build_sid_to_item(semantic_ids)
    sid2item_single, sid2item_multi = build_sid_to_item_tables(semantic_ids)

    allowed_tokens, next_states, branch_masks = build_prefix_branch_tables(
        sid2item,
        code_offset=model.CODE_OFFSET,
    )
    allowed_tokens, next_states, branch_masks = move_branch_tables_to_device(
        allowed_tokens,
        next_states,
        branch_masks,
        torch.device(device),
    )

    return (
        model,
        config,
        data,
        semantic_ids,
        sid2item,
        sid2item_single,
        sid2item_multi,
        allowed_tokens,
        next_states,
        branch_masks,
    )


@torch.inference_mode()
def recommend_next_items(
    model: CausalTransformer,
    history_items: list[int],
    semantic_ids: np.ndarray,
    sid2item_single,
    sid2item_multi,
    allowed_tokens,
    next_states,
    branch_masks,
    max_seq_len: int,
    beam_size: int,
    device: str,
    user_id: int = 0,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = build_model_input(
        history_items=history_items,
        semantic_ids=semantic_ids,
        max_seq_len=max_seq_len,
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
        amp_enabled=(device == "cuda" and torch.cuda.is_bf16_supported()),
        amp_dtype=(torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None),
    )

    candidates = beam_to_candidate(
        beams=beams,
        sid2item_single=sid2item_single,
        sid2item_multi=sid2item_multi,
        code_offset=model.CODE_OFFSET,
    )[0]
    raw_codes = beams[0].detach().to("cpu") - model.CODE_OFFSET

    return candidates, beams[0], raw_codes


def format_item_lines(
    item_ids: list[int],
    data: dict[str, Any],
    limit: int | None = None,
    show_titles: bool = True,
) -> list[str]:
    lines = []
    selected = item_ids if limit is None else item_ids[:limit]
    for idx, item_id in enumerate(selected, start=1):
        if show_titles:
            lines.append(f"{idx:2d}. {item_id} | {get_item_title(data, item_id)}")
        else:
            lines.append(f"{idx:2d}. {item_id}")
    return lines


def format_inference_output(
    data: dict[str, Any],
    user_id: int | None,
    split: str,
    history_items: list[int],
    target_item: int | None,
    candidates: list[int],
    beams: torch.Tensor,
    raw_codes: torch.Tensor,
    sid2item_single,
    sid2item_multi,
    topk: int,
    max_beams_to_print: int,
    show_titles: bool,
    code_offset: int,
) -> str:
    lines = []

    if user_id is not None:
        raw_user = data.get("id2user", {}).get(user_id, user_id)
        lines.append(f"User: {user_id} (raw={raw_user})")
        lines.append(f"Split: {split}")
    else:
        lines.append("User: manual_history")

    lines.append("")
    lines.append(f"History ({len(history_items)} items):")
    history_lines = format_item_lines(history_items, data, show_titles=show_titles)
    lines.extend(history_lines if history_lines else ["  <empty>"])

    if target_item is not None:
        lines.append("")
        if show_titles:
            lines.append(f"Target: {target_item} | {get_item_title(data, target_item)}")
        else:
            lines.append(f"Target: {target_item}")

        if target_item in candidates:
            target_rank = candidates.index(target_item) + 1
            lines.append(f"Target rank in candidates: {target_rank}")
        else:
            lines.append("Target rank in candidates: not found")

    lines.append("")
    lines.append(f"Top-{topk} recommended items:")
    rec_lines = format_item_lines(candidates, data, limit=topk, show_titles=show_titles)
    lines.extend(rec_lines if rec_lines else ["  <empty>"])

    lines.append("")
    lines.append(f"Top {min(max_beams_to_print, beams.size(0))} beams:")
    beams_cpu = beams.detach().to("cpu")
    raw_codes_cpu = raw_codes.detach().to("cpu")

    for beam_idx in range(min(max_beams_to_print, beams_cpu.size(0))):
        token_ids = beams_cpu[beam_idx].tolist()
        codes = raw_codes_cpu[beam_idx].tolist()
        sid = tuple(codes)
        single_item = sid2item_single.get(sid)
        if single_item is not None:
            items = [single_item]
        else:
            items = sid2item_multi.get(sid, [])

        lines.append(f"beam {beam_idx:2d}:")
        lines.append(f"  token_ids = {token_ids}")
        lines.append(f"  raw_codes = {codes}")
        if items:
            if show_titles:
                item_desc = [f"{item_id} | {get_item_title(data, item_id)}" for item_id in items[:5]]
            else:
                item_desc = [str(item_id) for item_id in items[:5]]
            lines.append(f"  items     = {item_desc}")
        else:
            lines.append("  items     = []")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--history", type=str, default=None, help="comma-separated item ids, e.g. 12,45,90")
    parser.add_argument("--user_id", type=int, default=None, help="real user id from beauty.pkl")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--beam_size", type=int, default=40)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_beams_to_print", type=int, default=5)
    parser.add_argument("--hide_titles", action="store_true", help="hide item titles in output")
    args = parser.parse_args()

    if (args.history is None) == (args.user_id is None):
        raise ValueError("Provide exactly one of --history or --user_id")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    (
        model,
        config,
        data,
        semantic_ids,
        sid2item,
        sid2item_single,
        sid2item_multi,
        allowed_tokens,
        next_states,
        branch_masks,
    ) = load_model_and_tables(
        args.checkpoint,
        device,
    )

    if args.history is not None:
        history_items = [int(x) for x in args.history.split(",") if x.strip()]
        target_item = None
        user_id = None
    else:
        history_items, target_item = get_user_history_for_inference(data, args.user_id, args.split)
        user_id = args.user_id

    candidates, beams, raw_codes = recommend_next_items(
        model=model,
        history_items=history_items,
        semantic_ids=semantic_ids,
        sid2item_single=sid2item_single,
        sid2item_multi=sid2item_multi,
        allowed_tokens=allowed_tokens,
        next_states=next_states,
        branch_masks=branch_masks,
        max_seq_len=config["max_seq_len"],
        beam_size=args.beam_size,
        device=device,
        user_id=(0 if user_id is None else user_id),
    )

    output = format_inference_output(
        data=data,
        user_id=user_id,
        split=args.split,
        history_items=history_items,
        target_item=target_item,
        candidates=candidates,
        beams=beams,
        raw_codes=raw_codes,
        sid2item_single=sid2item_single,
        sid2item_multi=sid2item_multi,
        topk=args.topk,
        max_beams_to_print=args.max_beams_to_print,
        show_titles=not args.hide_titles,
        code_offset=model.CODE_OFFSET,
    )
    print(output)


if __name__ == "__main__":
    main()
