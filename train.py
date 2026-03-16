import os
import json
import pickle
import time
import argparse
from pathlib import Path
import torch
import numpy as np
import swanlab
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from Amazon_Dataset import get_rec_loaders
from evaluate import build_sid_to_item, build_sid_to_item_tables, evaluate, print_metrics
from models.Transformer import CausalTransformer


CONFIG = {
    # 数据路径
    "data_path":       "datasets/processed/beauty.pkl",
    "sid_path":        "datasets/processed/semantic_ids.npy",
    "output_dir":      "checkpoints/rec",
    "save_every":      1,    # 每多少个 epoch 保存一次 latest ckpt
    "resume":          False, # 如有 latest.ckpt 自动断点续训
    "every_epoch":     5,

    # 模型超参数
    "d_model":         128,
    "num_head":           4,
    "num_layers":      4,
    "dim_feedforward": 512,
    "dropout_rate":         0.1,
    "num_rq_layers":   4,
    "codebook_size":   256,
    "use_user_token":  False,
    "target_loss_weights": [0.4, 0.3, 0.2, 0.1],
    "hierarchical_attention_enabled": True,
    "attention_layout": ["intra", "original", "original", "cross"],
    "max_seq_len":     50,
    "use_sliding_window": True,
    "sliding_window_mode": "sample_per_epoch",
    "window_size": 20,
    "min_seq_len": 2,
    "windows_per_user": 2,

    # 训练超参数
    "batch_size":      256,
    "epochs":          80,
    "lr":              3e-4,
    "min_lr":          3e-5,
    "weight_decay":    5e-5,
    "warmup_epochs":   1,
    "patience":        10,
    "amp_enabled":     True,
    "amp_dtype":       "auto",

    # 评估
    "beam_size":       40,
    "beam_schedule":   [40, 40, 40, 40],
    "train_eval_beam_size": 10,
    "train_eval_beam_schedule": [10, 10, 10, 10],
    "train_eval_topk": [1, 5, 10],
    "topk":            [1, 5, 10, 20, 40],
    
    "device":           'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'num_workers':      os.cpu_count(),
    'seed':             42
}


def parse_bool_arg(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def parse_int_list_arg(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    return values if values else None


def parse_str_list_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [x.strip() for x in value.split(",") if x.strip()]
    return values if values else None


def build_train_config_from_cli(base_config: dict) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Override resume flag: true/false")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--every_epoch", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--train_eval_beam_size", type=int, default=None)
    parser.add_argument("--beam_schedule", type=str, default=None, help="comma-separated beam schedule")
    parser.add_argument("--train_eval_beam_schedule", type=str, default=None, help="comma-separated train-eval beam schedule")
    parser.add_argument("--hierarchical_attention_enabled", type=str, default=None, help="Override hierarchical attention flag: true/false")
    parser.add_argument("--attention_layout", type=str, default=None, help="comma-separated attention layout, e.g. intra,original,original,cross")
    args = parser.parse_args()

    config = dict(base_config)

    if args.resume is not None:
        config["resume"] = parse_bool_arg(args.resume)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.every_epoch is not None:
        config["every_epoch"] = args.every_epoch
    if args.beam_size is not None:
        config["beam_size"] = args.beam_size
    if args.train_eval_beam_size is not None:
        config["train_eval_beam_size"] = args.train_eval_beam_size
    if args.hierarchical_attention_enabled is not None:
        config["hierarchical_attention_enabled"] = parse_bool_arg(args.hierarchical_attention_enabled)

    beam_schedule = parse_int_list_arg(args.beam_schedule)
    if beam_schedule is not None:
        config["beam_schedule"] = beam_schedule

    train_eval_beam_schedule = parse_int_list_arg(args.train_eval_beam_schedule)
    if train_eval_beam_schedule is not None:
        config["train_eval_beam_schedule"] = train_eval_beam_schedule

    attention_layout = parse_str_list_arg(args.attention_layout)
    if attention_layout is not None:
        config["attention_layout"] = attention_layout

    return config


def print_parameter_summary(model:torch.nn.Module) -> None:
    total_params = 0
    trainable_params = 0
    
    print('\nParameter Summary')
    print('-'*60)
    
    module_param_counts = {}
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        top_module = name.split('.')[0]
        module_param_counts[top_module] = module_param_counts.get(top_module, 0) + num_params

    for module_name, count in sorted(module_param_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{module_name:20s} {count:12,d}")

    print('-'*60)
    print(f"{'total':20s} {total_params:12,d}")
    print(f"{'trainable':20s} {trainable_params:12,d}")
    print('-'*60)


def warn_if_eval_beam_too_narrow(beam_schedule: list[int] | None, topk: list[int], label: str) -> None:
    if not beam_schedule or not topk:
        return

    final_beam = beam_schedule[-1]
    max_k = max(topk)
    if final_beam < max_k:
        print(
            f"[WARN] {label} final beam width={final_beam} < max(topk)={max_k}. "
            "Top-k metrics may be underestimated because too few final SID paths are kept."
        )

class WarmupCosineScheduler:
    """
    学习率调度：线性预热 + 余弦退火
    
    - 预热阶段(epoch <= warmup_epochs):
        lr 随epoch数，线性增长到 base_lr
    - 余弦退火阶段
        lr 从base_lr平滑降低到 min_lr
    -为什么需要预热：
        训练初期参数随机，梯度方向不稳定
        直接用大 lr 容易导致 loss 爆炸
        预热让模型先找到大致方向再加速
    """
    def __init__(self, optimizer:torch.optim.Optimizer,
                 warmup_epochs:int,
                 total_epochs:int,
                 base_lr:float,
                 min_lr:float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
    
    def step(self, epoch: int) -> float:
        """
        epoch 从 1 开始，返回当前 lr
        """
        if self.total_epochs <= self.warmup_epochs:
            lr = self.base_lr
        elif epoch <= self.warmup_epochs:
            warmup_denom = max(self.warmup_epochs, 1)
            lr = self.base_lr * epoch / warmup_denom
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


def resolve_amp_settings(config: dict, device: str) -> tuple[bool, torch.dtype | None, bool]:
    """
    训练期 AMP 只在 CUDA 上启用。
    默认优先 bf16，不支持时回退到 fp16；只有 fp16 需要 GradScaler。
    """
    amp_enabled = bool(config.get("amp_enabled", True)) and device == "cuda"
    if not amp_enabled:
        return False, None, False

    amp_dtype = str(config.get("amp_dtype", "auto")).lower()
    bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    if amp_dtype == "auto":
        chosen_dtype = torch.bfloat16 if bf16_supported else torch.float16
    elif amp_dtype == "bf16":
        if not bf16_supported:
            raise ValueError("amp_dtype='bf16' requires CUDA bf16 support")
        chosen_dtype = torch.bfloat16
    elif amp_dtype == "fp16":
        chosen_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported amp_dtype: {config.get('amp_dtype')}")

    grad_scaler_enabled = chosen_dtype == torch.float16
    return True, chosen_dtype, grad_scaler_enabled
    

def train_one_epoch(
    model: CausalTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
)->dict[str, float | list[float]]:
    """
    训练一个epoch, 返回平均loss
    """
    model.train()
    total_loss = 0.0
    total_exact_acc = 0.0
    total_token_acc = 0.0
    total_layer_acc = torch.zeros(model.num_rq_layers, dtype=torch.float64)
    num_batches = len(loader)
    autocast_device_type = "cuda" if device == "cuda" else "cpu"
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=False)
    for batch in pbar:
        user_ids = batch['user_id'].to(device, non_blocking=True)   # [B]
        input_ids = batch['input_ids'].to(device, non_blocking=True)   # [B, T]
        attention_mask = batch['attention_mask'].to(device, non_blocking=True) # [B, T]
        target_ids = batch['target_ids'].to(device, non_blocking=True) # [B, L]
        
        optimizer.zero_grad(set_to_none=True)

        # forward + calculate loss
        with torch.autocast(
            device_type=autocast_device_type,
            dtype=amp_dtype,
            enabled=amp_enabled
        ):
            outputs = model.compute_loss(input_ids, attention_mask, target_ids, user_ids)
            loss = outputs["loss"]
            exact_acc = outputs["exact_acc"]
            token_acc = outputs["token_acc"]
            layer_acc = outputs["layer_acc"]
        
        # backward
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 截断梯度，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        total_exact_acc += exact_acc.item()
        total_token_acc += token_acc.item()
        total_layer_acc += layer_acc.detach().to("cpu", dtype=torch.float64)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'exact_acc': f'{exact_acc.item():.4f}',
            'token_acc': f'{token_acc.item():.4f}',
        })
        
    avg_layer_acc = (total_layer_acc / num_batches).tolist()
    return {
        "loss": total_loss / num_batches,
        "exact_acc": total_exact_acc / num_batches,
        "token_acc": total_token_acc / num_batches,
        "layer_acc": avg_layer_acc,
    }


def build_timestamped_ckpt_path(output_dir: Path, prefix: str, epoch: int) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_epoch{epoch:03d}_{timestamp}.pt"


def train_rec(config:dict = CONFIG):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = config['device']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = output_dir / 'latest.pt'
    best_ckpt_path   = output_dir / 'best_model.pt'
    latest_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Using {device}..")

    amp_enabled, amp_dtype, grad_scaler_enabled = resolve_amp_settings(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=grad_scaler_enabled)
    amp_dtype_name = "fp32" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")
    print(
        f"AMP | amp_enabled={amp_enabled} "
        f"amp_dtype={amp_dtype_name} "
        f"grad_scaler_enabled={scaler.is_enabled()}"
    )
    
    print("Loading Data..")
    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    
    semantic_ids = np.load(config['sid_path'])
    if semantic_ids.ndim != 2:
        raise ValueError(f"semantic_ids must be 2D [num_items, num_rq_layers], got shape={semantic_ids.shape}")

    sid_num_rq_layers = int(semantic_ids.shape[1])
    cfg_num_rq_layers = int(config.get('num_rq_layers', sid_num_rq_layers))
    if cfg_num_rq_layers != sid_num_rq_layers:
        print(
            f"[WARN] num_rq_layers from config={cfg_num_rq_layers} "
            f"!= sid layers={sid_num_rq_layers}, using sid layers."
        )
    num_rq_layers = sid_num_rq_layers

    target_loss_weights = config.get('target_loss_weights')
    if target_loss_weights is None or len(target_loss_weights) != num_rq_layers:
        print(
            f"[WARN] target_loss_weights length mismatch "
            f"(got {None if target_loss_weights is None else len(target_loss_weights)}; "
            f"expect {num_rq_layers}), using uniform weights."
        )
        target_loss_weights = [1.0] * num_rq_layers

    sid2item = build_sid_to_item(semantic_ids)
    sid2item_single, sid2item_multi = build_sid_to_item_tables(semantic_ids)
    
    print("Building Dataloader")
    train_loader, val_loader, test_loader, vocab_size = get_rec_loaders(
        data=data,
        semantic_ids=semantic_ids,
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        num_rq_layers=num_rq_layers,
        num_workers=config['num_workers'],
        use_sliding_window=config.get('use_sliding_window', True),
        sliding_window_mode=config.get('sliding_window_mode', 'all'),
        window_size=config.get('window_size', 20),
        min_seq_len=config.get('min_seq_len', 2),
        windows_per_user=config.get('windows_per_user', 2),
        seed=config['seed'],
    )
    
    print("Initialize Model")
    # Teacher Forcing 时会拼入 target prefix（num_rq_layers-1 个 token）
    # 所以 max_seq_len 要留出这部分空间
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
    print(
        "Attention layout | "
        f"enabled={config.get('hierarchical_attention_enabled', False)} "
        f"layout={model.attention_layout}"
    )
    print_parameter_summary(model)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['epochs'],
        base_lr=config['lr'],
        min_lr=config.get('min_lr', 1e-6)
    )

    # 断点续训准备
    start_epoch = 1
    best_val_ndcg = 0.0
    patience_count = 0
    history = []

    if config.get('resume', False) and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if scaler.is_enabled() and ckpt.get('scaler_state') is not None:
            scaler.load_state_dict(ckpt['scaler_state'])
        best_val_ndcg = ckpt.get('best_val_ndcg', 0.0)
        patience_count = ckpt.get('patience_count', 0)
        history = ckpt.get('history', [])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resume from {latest_ckpt_path} (epoch {ckpt['epoch']})")
    else:
        print("Training from scratch..")
    
    # 初始化 SwanLab
    swanlab.init(
        project="Generated_Retrieval",
        experiment_name="CausalTransformer",
        config={
            **config,
            "amp_enabled_runtime": amp_enabled,
            "amp_dtype_runtime": amp_dtype_name,
            "grad_scaler_enabled_runtime": scaler.is_enabled(),
        }
    )

    print("Start Training..")

    # 早停监控的主指标
    # 优先用 NDCG@10, 如果 topk 里面没有 10 就用最大的 K 
    train_eval_topk = config.get('train_eval_topk', config['topk'])
    monitor_k = 10 if 10 in train_eval_topk else max(train_eval_topk)
    print(
        f"Train-eval beam | size={config.get('train_eval_beam_size', config['beam_size'])} "
        f"schedule={config.get('train_eval_beam_schedule')}"
    )
    print(
        f"Test beam       | size={config['beam_size']} "
        f"schedule={config.get('beam_schedule')}"
    )
    warn_if_eval_beam_too_narrow(config.get('train_eval_beam_schedule'), train_eval_topk, "train-eval")
    warn_if_eval_beam_too_narrow(config.get('beam_schedule'), config['topk'], "test")

    for epoch in range(start_epoch, config['epochs'] + 1):
        # update learning rate
        current_lr = scheduler.step(epoch)

        if hasattr(train_loader.dataset, "resample_samples"):
            train_loader.dataset.resample_samples(epoch)
        train_num_samples = len(train_loader.dataset)
        print(f"Epoch {epoch:3d} train_samples={train_num_samples}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        train_loss = float(train_metrics["loss"])
        train_exact_acc = float(train_metrics["exact_acc"])
        train_token_acc = float(train_metrics["token_acc"])
        train_layer_acc = [float(v) for v in train_metrics["layer_acc"]]
        
        if epoch % config['every_epoch'] == 0 or epoch == config['epochs']:
            eval_start = time.perf_counter()
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                sid2item=sid2item,
                sid2item_single=sid2item_single,
                sid2item_multi=sid2item_multi,
                topk=train_eval_topk,
                beam_size=config.get('train_eval_beam_size', config['beam_size']),
                device=device,
                beam_schedule=config.get('train_eval_beam_schedule'),
                split='val'
            )
            eval_time = time.perf_counter() - eval_start
            
            print(
                f"\nEpoch: {epoch:3d}/{config['epochs']} lr={current_lr:.2e} "
                f"loss={train_loss:.4f} exact_acc={train_exact_acc:.4f} token_acc={train_token_acc:.4f}"
            )
            print(
                "Train layer_acc="
                + ", ".join(f"c{idx}={value:.4f}" for idx, value in enumerate(train_layer_acc))
            )
            print_metrics(val_metrics, train_eval_topk, prefix='Val')
            print(f"Val decode_time={eval_time:.2f}s")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_exact_acc,
                'train_exact_acc': train_exact_acc,
                'train_token_acc': train_token_acc,
                'train_layer_acc': train_layer_acc,
                'train_num_samples': train_num_samples,
                'lr': current_lr,
                'amp_enabled': amp_enabled,
                'amp_dtype': amp_dtype_name,
                'grad_scaler_enabled': scaler.is_enabled(),
                'val_decode_time_sec': eval_time,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            # SwanLab 记录
            train_log_payload = {
                'train/loss': train_loss,
                'train/acc': train_exact_acc,
                'train/exact_acc': train_exact_acc,
                'train/token_acc': train_token_acc,
                'train/lr': current_lr,
                'train/num_samples': train_num_samples,
                'train/amp_enabled': float(amp_enabled),
                'train/grad_scaler_enabled': float(scaler.is_enabled()),
                'val/decode_time_sec': eval_time,
                **{f'val/{k}': v for k, v in val_metrics.items()}
            }
            for idx, value in enumerate(train_layer_acc):
                train_log_payload[f'train/layer_acc_c{idx}'] = value
            swanlab.log(train_log_payload, step=epoch)
            
            # Check Early Stopping
            val_ndcg = val_metrics[f"NDCG@{monitor_k}"]
            
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                patience_count = 0

                best_state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict() if scaler.is_enabled() else None,
                    'val_metrics': val_metrics,
                    'config': config
                }
                torch.save(best_state, best_ckpt_path)
                best_snapshot_path = build_timestamped_ckpt_path(output_dir, "best_model", epoch)
                torch.save(best_state, best_snapshot_path)
                print(f"Saved Best Model(NDCG@{monitor_k})={val_ndcg:.4f}")
                print(f"Saved best snapshot to {best_snapshot_path}")
            else:
                patience_count += 1
                print(f"  patience: {patience_count}/{config['patience']}")
                if patience_count >= config["patience"]:
                    print(f"\n早停：NDCG@{monitor_k} 连续 "
                        f"{config['patience']} epoch 未提升")
                    break

            # 保存 latest 以便断点续训（放在早停判断后，确保状态是最新）
            if (epoch % config['save_every']) == 0:
                latest_state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict() if scaler.is_enabled() else None,
                    'best_val_ndcg': best_val_ndcg,
                    'patience_count': patience_count,
                    'history': history,
                    'config': config
                }
                torch.save(latest_state, latest_ckpt_path)
                latest_snapshot_path = build_timestamped_ckpt_path(output_dir, "latest", epoch)
                torch.save(latest_state, latest_snapshot_path)
                print(f"Saved latest checkpoint to {latest_ckpt_path}")
                print(f"Saved latest snapshot to {latest_snapshot_path}")
        else:
            # 非评估轮次只打印 loss
            train_log_payload = {
                'train/loss': train_loss,
                'train/acc': train_exact_acc,
                'train/exact_acc': train_exact_acc,
                'train/token_acc': train_token_acc,
                'train/lr': current_lr,
                'train/num_samples': train_num_samples,
                'train/amp_enabled': float(amp_enabled),
                'train/grad_scaler_enabled': float(scaler.is_enabled()),
            }
            for idx, value in enumerate(train_layer_acc):
                train_log_payload[f'train/layer_acc_c{idx}'] = value
            swanlab.log(train_log_payload, step=epoch)
            print(
                f"Epoch {epoch:3d} | loss={train_loss:.4f} "
                f"exact_acc={train_exact_acc:.4f} token_acc={train_token_acc:.4f} "
                f"samples={train_num_samples} | (skip eval)"
            )
    
    print("Start Evaluating..")
    eval_ckpt_path = best_ckpt_path if best_ckpt_path.exists() else latest_ckpt_path if latest_ckpt_path.exists() else None
    if eval_ckpt_path is None:
        print("No checkpoint found for evaluation. Skipping test phase.")
        return model, {}

    ckpt = torch.load(eval_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    best_val_ndcg = ckpt.get('best_val_ndcg', best_val_ndcg)
    best_epoch = ckpt.get('epoch', 'unknown')
    print(f"Loading Model from {eval_ckpt_path} (Epoch {best_epoch}, val NDCG@{monitor_k}={best_val_ndcg:.4f})..")
            
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        sid2item=sid2item,
        sid2item_single=sid2item_single,
        sid2item_multi=sid2item_multi,
        topk=config['topk'],
        beam_size=config['beam_size'],
        device=device,
        beam_schedule=config.get('beam_schedule'),
        split='test'
    )
    
    print('\nFinal Results:')
    print_metrics(test_metrics, config['topk'], prefix='Test')

    # 记录测试集最终指标
    swanlab.log({f'test/{k}': v for k, v in test_metrics.items()})
    swanlab.finish()
    
    results = {
        'history': history,
        'test_metrics': test_metrics,
        'best_epoch': ckpt['epoch'],
        'best_val_ndcg': best_val_ndcg,
        'config': config
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved Results: {output_dir / 'results.json'}")
    
    print("Training Finished!")
    return model, test_metrics

if __name__ == "__main__":
    train_rec(build_train_config_from_cli(CONFIG))
    
    
        
