from tqdm.auto import tqdm
import os
import pickle
from pathlib import Path
from Amazon_Dataset import ItemEmbeddingDataset, get_rqvae_loaders
from models.RQVAE import RQVAE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Tuple, List, Dict
import swanlab


CONFIG = {
    # 数据
    "data_path":      "datasets/processed/beauty.pkl",
    "emb_path":       "datasets/processed/item_embeddings.npy",
    "output_dir":     "checkpoints/rqvae",
    "resume":         False,
    "save_every":     1,
    'sid_path':       "datasets/processed/semantic_ids.npy",
    
    # 模型
    "input_dim":      384,    # Sentence-BERT 输出维度
    "hidden_dim":     256,
    "latent_dim":     64,
    "codebook_size":  256,
    "num_rq_layers":  3,
    "decay":          0.99,
    "commitment_weight": 0.10,
    "dropout":        0.1,
    
    # 训练
    "batch_size":     512,
    "epochs":         100,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "val_ratio":      0.1,
    "patience":       10,     # 早停：val loss 不下降多少 epoch 就停
    "min_delta":      1e-4,
    
    # 硬件
    "device":         "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu",
    "num_workers":    os.cpu_count(),
    "seed":           42,
}


def reconstruction_loss(x_recon: torch.tensor, x:torch.tensor)->torch.tensor:
    """
    余弦相似度 reconstruction loss
    - 为什么用余弦而不是 MSE：
        Sentence-BERT 的 embedding 已经归一化
        语义相似性体现在方向上，不是绝对距离
        余弦损失对 embedding 的 scale 不敏感
    loss = 1 - cosine_similarity
    - 完美重建时 loss = 0
    - 完全相反时 loss = 2
    """
    loss = 1 - F.cosine_similarity(x_recon, x)
    return loss

def total_loss(x_recon:torch.tensor,
               x:torch.tensor,
               vq_loss:torch.tensor, 
               recon_weight:float = 1.0) -> Tuple[torch.tensor, dict]:
    """
    总损失 = 重建损失 + 量化损失
    
    返回 loss 和各分量的 dict（用于日志）
    """
    recon_loss = reconstruction_loss(x_recon, x).mean()
    total_loss = recon_weight*recon_loss + vq_loss
    
    return total_loss, {
        "loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "vq_loss": vq_loss.item()
    }
    
    
def extract_embedding(
    item_texts:list[str],
    save_path: Path,
    batch_size:int = 256,
    device=CONFIG['device']
) -> np.ndarray:
    """
    用 Sentence-BERT 提取 item 文本 embedding
    
    为什么在训练 RQ-VAE 之前单独提取：
      Sentence-BERT 提取是一次性的，结果不会变
      提取完存到磁盘，训练时直接读取
      避免每个 epoch 都重新提取（浪费时间）
    """
    save_path = Path(save_path)
    if save_path.exists():
        print(f"加载已有 embedding: {save_path}")
        return np.load(save_path)
    
    print("Extracting Item Embedding (Sentence-BERT)...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to(device)
    
    embeddings = model.encode(
        item_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # L2归一化，让余弦相似度等价于点积
    )
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embeddings)
    print(f"保存 embedding: {save_path}, shape: {embeddings.shape}")
    
    return embeddings


def train_one_epoch(
    model: RQVAE,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device
)->dict:
    """
    训练一个epoch
    返回各损失的平均值
    """
    model.train()
    
    total_metrics = {
        "loss": 0.,
        "recon_loss": 0.,
        "vq_loss": 0.
    }
    
    for batch in loader:
        x = batch.to(device)    # [B, input_dim]
        
        # forward
        x_recon, vq_loss, codes = model(x)
        
        # Calculate Loss
        loss, metrics = total_loss(x_recon, x, vq_loss)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # gradient descent
        optimizer.step()
        
        # 累计损失
        for k, v in metrics.items():
            total_metrics[k] += float(v) / len(loader)
    
    return total_metrics


@torch.inference_mode()
def validation(model:RQVAE,
             loader: torch.utils.data.DataLoader,
             device) -> dict:
    """
    验证集评估
    额外计算：码本利用率 + 语义ID碰撞率
    """
    model.eval()
    
    total_metrics = {
        'loss':0.,
        'recon_loss':0.,
        'vq_loss':0.
        }
    all_codes = []  # 收集所有codes，用于计算碰撞率
    
    for batch in loader:
        x = batch.to(device)
        x_recon, vq_loss, codes = model(x)
        loss, metrics = total_loss(x_recon, x, vq_loss)
        
        for k, v in metrics.items():
            total_metrics[k] += float(v)/len(loader)
        
        all_codes.append(codes.cpu().numpy())
    
    # 计算碰撞率
    all_codes = np.concatenate(all_codes, axis=0)   # [N, num_layers]
    unique_ids = len(set(map(tuple, all_codes.tolist())))
    total_ids = len(all_codes)
    collision_rate = 1 - unique_ids / total_ids
    
    # 码本利用率
    utilizations = model.rq.utilization_per_layer
    
    total_metrics['collision_rate'] = collision_rate
    total_metrics['utilization'] = np.mean(utilizations)
    
    return total_metrics


@torch.inference_mode()
def generate_semantic_ids(
    model:RQVAE,
    embeddings:np.ndarray,
    device,
    batch_size:int = 1024
)->np.ndarray:
    """
    对所有item生成语义ID
    Returns:
        - Semantic IDs: [num_items, num_rq_layers]
    """
    model.eval()
    dataset = ItemEmbeddingDataset(embeddings)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False)
    all_codes = []
    
    for batch in loader:
        x = batch.to(device)
        codes = model.generate_semantic_ids(x)
        all_codes.append(codes.cpu().numpy())
    
    semantic_ids = np.concatenate(all_codes, axis=0) # [N, num_rq_layer]
    
    # 统计
    unique = len(set(map(tuple, semantic_ids.tolist())))
    print(f"生成语义ID: {len(semantic_ids)} 个item")
    print(f"  唯一ID数: {unique}")
    print(f"  碰撞率:   {1 - unique/len(semantic_ids):.2%}")
    
    return semantic_ids


def kmeans_init_codebooks(
    model:RQVAE,
    embeddings:np.ndarray,
    device:str
):
    """
    用KMeans初始化每一层的码本
    关键点：第 l 层在第 l-1 层的残差上做KMeans，不实在原始embedding上做
    
    原因：
      第0层码本负责捕捉粗粒度语义
      第1层码本负责捕捉第0层留下的残差
      如果都在原始 embedding 上做 KMeans
      三层码本会学到重复的信息
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    print("\nKMeans初始化码本..")
    model.eval()
    
    all_z = []
    dataset = ItemEmbeddingDataset(embeddings)
    loader = DataLoader(dataset,
                        batch_size=1024,
                        shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            z = model.encoder(batch.to(device))
            all_z.append(z.cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)   # [N, latent_dim]
    
    # 逐层初始化，在残差上做 KMeans
    residual = all_z.copy()

    for layer_idx, quantizer in enumerate(model.rq.quantizers):
        K = quantizer.codebook_size
        print(f"  第{layer_idx}层（K={K}）...")
        # KMeans
        n = len(residual)
        if n > 100000:
            km = MiniBatchKMeans(
                n_clusters=K, n_init=3,
                batch_size=4096, random_state=42
            )
        else:
            km = KMeans(
                n_clusters=K, n_init=10, random_state=42
            )
        km.fit(residual)
        centers = torch.FloatTensor(km.cluster_centers_).to(device)
        
        # 写入码本
        with torch.inference_mode():
            quantizer.codebook.copy_(centers)
            quantizer.ema_weight.copy_(centers)
            count = torch.FloatTensor(
                np.bincount(km.labels_, minlength=K).astype(np.float32)
            ).to(device)
            quantizer.ema_count.copy_(count.clamp(min=1))

        # 计算残差
        labels = km.labels_
        quantized = km.cluster_centers_[labels]
        residual = residual - quantized
        
        residual_norm = np.linalg.norm(residual, axis=-1).mean()
        utilization = (np.bincount(labels, minlength=K) > 0).mean()
        print(f"    利用率: {utilization:.1%}, 残差均值: {residual_norm:.4f}")
    
    print("KMeans 初始化完成\n")


def train_rqvae(config:dict = CONFIG):
    torch.manual_seed(config['seed'])
    device = config['device']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 SwanLab 记录
    swanlab.init(project="rqvae", name=f"train-rqvae-lr={config['lr']}-cw={config['commitment_weight']}", config=config)
    
    print(f"Using device: {device}")
    
    # 加载 数据 & Embedding
    print("Loading Data & Embedding..")
    with open (config['data_path'], 'rb') as f:
        data = pickle.load(f)
    
    embeddings = extract_embedding(
        data['item_texts'],
        save_path=Path(config['emb_path']),
        device=device
    )   # [num_items, input_dim] = [num_items, 384]
    
    print(f"\nembedding shape: {embeddings.shape}")
    
    train_loader, val_loader = get_rqvae_loaders(
        embeddings=embeddings,
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=config['num_workers'],
        seed=config['seed']
    )
    
    print("Initialize Model")
    model = RQVAE(
        input_dim=embeddings.shape[1],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        codebook_size=config['codebook_size'],
        num_layers=config['num_rq_layers'],
        decay=config['decay'],
        commitment_cost=config['commitment_weight'],
        dropout_rate=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")
    
    # KMeans 初始化码本
    kmeans_init_codebooks(model, embeddings, device)
    
    print("Start training ..")
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config['epochs'],
                                  eta_min=config['lr']*0.01)
    
    # Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    min_delta = config.get('min_delta', 0.0)
    best_model_path = output_dir/'best_model.pt'
    latest_model_path = output_dir/'latest.pt'

    start_epoch = 1
    if config.get('resume', False) and latest_model_path.exists():
        latest = torch.load(latest_model_path, map_location=device, weights_only=False)
        model.load_state_dict(latest['model_state'])
        optimizer.load_state_dict(latest['optimizer_state'])
        scheduler.load_state_dict(latest['scheduler_state'])
        best_val_loss = latest.get('best_val_loss', best_val_loss)
        patience_counter = latest.get('patience_counter', patience_counter)
        start_epoch = latest.get('epoch', 0) + 1
        print(f"Resume from {latest_model_path} (epoch {start_epoch-1})")
    
    for epoch in range(start_epoch, config['epochs'] + 1):
        # Train
        train_metrics = train_one_epoch(model,
                                        train_loader,
                                        optimizer,
                                        device)
        # Validation
        val_metrics = validation(model,
                                 val_loader,
                                 device)
        scheduler.step()
        
        # printing log
        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"recon={train_metrics['recon_loss']:.4f} "
            f"vq={train_metrics['vq_loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"collision={val_metrics['collision_rate']:.2%} "
            f"util={val_metrics['utilization']:.1%}"
        )

        # 记录指标到 SwanLab
        current_lr = optimizer.param_groups[0]['lr']
        swanlab.log({
            "train/loss": train_metrics['loss'],
            "train/recon": train_metrics['recon_loss'],
            "train/vq": train_metrics['vq_loss'],
            "val/loss": val_metrics['loss'],
            "val/collision_rate": val_metrics['collision_rate'],
            "val/utilization": val_metrics['utilization'],
            "lr": current_lr,
        }, step=epoch)
        
        # Check early stopping
        if val_metrics['loss'] < (best_val_loss - min_delta):
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n早停：val_loss 连续 {config['patience']} epoch 不下降")
                break

        if (epoch % config.get('save_every', 1)) == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config,
            }, latest_model_path)
        
    print("Generating Semantic IDs")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    
    semantic_ids = generate_semantic_ids(model, embeddings, device)
    
    # 保存 SID
    sid_path = Path(config['sid_path'])
    np.save(sid_path, semantic_ids)
    print(f"语义ID 保存到: {sid_path}")
    print(f"Shape: {semantic_ids.shape}")
    
    print("\n✅ RQ-VAE 训练完成！")
    return model, semantic_ids

    
if __name__ == "__main__":
    train_rqvae()