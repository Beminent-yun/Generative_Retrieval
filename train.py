import os
import json
import pickle
from pathlib import Path
import torch
import numpy as np
import swanlab
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Amazon_Dataset import get_rec_loaders
from evaluate import build_sid_to_item, evaluate, print_metrics
from models.Transformer import CausalTransformer


CONFIG = {
    # 数据路径
    "data_path":       "datasets/processed/beauty.pkl",
    "sid_path":        "datasets/processed/semantic_ids_rkmeans.npy",
    "output_dir":      "checkpoints/rec",
    "save_every":      1,    # 每多少个 epoch 保存一次 latest ckpt
    "resume":          True, # 如有 latest.ckpt 自动断点续训
    "every_epoch":     5,

    # 模型超参数
    "d_model":         128,
    "num_head":           4,
    "num_layers":      4,
    "dim_feedforward": 512,
    "dropout_rate":         0.1,
    "num_rq_layers":   4,
    "codebook_size":   256,
    "use_user_token":  True,
    "target_loss_weights": [0.4, 0.3, 0.2, 0.1],
    "max_seq_len":     50,
    "use_sliding_window": True,
    "window_size": 20,
    "min_seq_len": 2,

    # 训练超参数
    "batch_size":      256,
    "epochs":          80,
    "lr":              3e-4,
    "min_lr":          3e-5,
    "weight_decay":    5e-5,
    "warmup_epochs":   1,
    "patience":        10,

    # 评估
    "beam_size":       20,
    "topk":            [1, 5, 10, 20, 40],
    
    "device":           'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'num_workers':      os.cpu_count(),
    'seed':             42
}


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
    

def train_one_epoch(
    model: CausalTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
)->tuple[float, float]:
    """
    训练一个epoch, 返回平均loss
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(loader)
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=False)
    for batch in pbar:
        user_ids = batch['user_id'].to(device)   # [B]
        input_ids = batch['input_ids'].to(device)   # [B, T]
        attention_mask = batch['attention_mask'].to(device) # [B, T]
        target_ids = batch['target_ids'].to(device) # [B, L]
        
        # forward + calculate loss
        outputs = model.compute_loss(input_ids, attention_mask, target_ids, user_ids)
        loss = outputs["loss"]
        acc = outputs["acc"]
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        # 截断梯度，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})
        
    return total_loss / num_batches, total_acc / num_batches


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
    
    print("Building Dataloader")
    train_loader, val_loader, test_loader, vocab_size = get_rec_loaders(
        data=data,
        semantic_ids=semantic_ids,
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        num_rq_layers=num_rq_layers,
        num_workers=config['num_workers'],
        use_sliding_window=config.get('use_sliding_window', True),
        window_size=config.get('window_size', 20),
        min_seq_len=config.get('min_seq_len', 2),
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
        target_loss_weights=target_loss_weights
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'总参数量：{total_params}')
    
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
        config=config
    )

    print("Start Training..")

    # 早停监控的主指标
    # 优先用 NDCG@10, 如果 topk 里面没有 10 就用最大的 K 
    monitor_k = 10 if 10 in config['topk'] else max(config['topk'])

    for epoch in range(start_epoch, config['epochs'] + 1):
        # update learning rate
        current_lr = scheduler.step(epoch)
        
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        if epoch % config['every_epoch'] == 0 or epoch == config['epochs']:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                sid2item=sid2item,
                topk=config['topk'],
                beam_size=config['beam_size'],
                device=device,
                split='val'
            )
            
            print(f"\nEpoch: {epoch:3d}/{config['epochs']} lr={current_lr:.2e} loss={train_loss:.4f} acc={train_acc:.4f}")
            print_metrics(val_metrics, config['topk'], prefix='Val')
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'lr': current_lr,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            # SwanLab 记录
            swanlab.log({
                'train/loss': train_loss,
                'train/acc': train_acc,
                'train/lr': current_lr,
                **{f'val/{k}': v for k, v in val_metrics.items()}
            }, step=epoch)
            
            # Check Early Stopping
            val_ndcg = val_metrics[f"NDCG@{monitor_k}"]
            
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                patience_count = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config
                }, best_ckpt_path)
                print(f"Saved Best Model(NDCG@{monitor_k})={val_ndcg:.4f}")
            else:
                patience_count += 1
                print(f"  patience: {patience_count}/{config['patience']}")
                if patience_count >= config["patience"]:
                    print(f"\n早停：NDCG@{monitor_k} 连续 "
                        f"{config['patience']} epoch 未提升")
                    break

            # 保存 latest 以便断点续训（放在早停判断后，确保状态是最新）
            if (epoch % config['save_every']) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_ndcg': best_val_ndcg,
                    'patience_count': patience_count,
                    'history': history,
                    'config': config
                }, latest_ckpt_path)
                print(f"Saved latest checkpoint to {latest_ckpt_path}")
        else:
            # 非评估轮次只打印 loss
            print(f"Epoch {epoch:3d} | loss={train_loss:.4f} acc={train_acc:.4f} | (skip eval)")
    
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
        topk=config['topk'],
        beam_size=config['beam_size'],
        device=device,
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
    train_rec()
    
    
        