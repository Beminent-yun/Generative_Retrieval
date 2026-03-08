from pathlib import Path
import pickle
from sklearn.cluster import MiniBatchKMeans
import time
import numpy as np
from typing import List
from torch import nn

class RKMeans_Tokenizer(nn.Module):
    """
    Residual Mini-Batch K-Means 语义ID编码器
    
    核心思路：
        逐层对残差做K-Means
        每层利用K-Means算法学习codebook(centroids矩阵)
        最终用每层最近邻的 centroid index 作为SID
    
    参数：
        num_layers:    残差层数，对应 SID 的长度（默认3）
        codebook_size: 每层的 centroid 数量（默认256）
        embed_dim: 输入 embedding 的维度
        normalize:     是否在每层对残差做 L2 归一化（默认True）
                        论文里 RK-Means 和 R-VQ 都做归一化以防 collapse
        batch_size:    Mini-Batch K-Means 的 batch size
        max_iter:      每层 K-Means 的最大迭代次数
        n_init:        K-Means 初始化次数（取最好的）
        random_state:  随机种子
    """
    def __init__(self, num_layers: int = 3,
                 codebook_size: int = 256,
                 embed_dim: int = 384,
                 normalize: bool = True,
                 batch_size:int = 4096,
                 max_iter: int = 1000,
                 n_init: int = 3,
                 random_state: int = 42):
        super().__init__()
        self.num_layers    = num_layers
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.normalize     = normalize
        self.batch_size    = batch_size
        self.max_iter      = max_iter
        self.n_init        = n_init
        self.random_state  = random_state

        # 每层的centroids: List[np.ndarray(codebook_size, embed_dim)]
        self.codebooks: List[np.ndarray] = []
        
        # 训练完后存储每个item的Semantic ID
        self._semantic_ids = None
        
        self._is_fitted = False  # 是否训练完毕
    
    def forward(self, embeddings: np.ndarray) -> 'RKMeans_Tokenizer':
        """
        逐层训练 K-Means
        
        embeddings: (num_items, embed_dim)
        
        训练过程：
            第一层：KMeans(embeddings) -> codebook_1
                    residual_1 = embeddings - codebook[optim_idx_1]
            第二层：KMeans(residual_1) -> codebook_2
                    residual_2 = residual_1 - codebook[optim_idx_2] 
            第三层：KMeans(residual_2) -> codebook_3    
        """
        print(f"RK-Means 训练开始")
        print(f"  item 数量:    {len(embeddings)}")
        print(f"  embedding 维度: {embeddings.shape[1]}")
        print(f"  残差层数:    {self.num_layers}")
        print(f"  codebook 大小: {self.codebook_size}")
        
        self.codebooks = []
        residual = embeddings.copy().astype(np.float32)
        
        for layer in range(self.num_layers):
            t0 = time.time()
            print(f"\n[第 {layer+1}/{self.num_layers} 层]")
            
            # 归一化残差(防止不同层之间的尺度差异)
            if self.normalize:
                norms = np.linalg.norm(residual, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8) # 防止除以0
                residual_normed = residual / norms
            else:
                residual_normed = residual
                
            # Mini-Batch K-Means
            
            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_size,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                compute_labels=True,
                verbose=0
            )
            kmeans.fit(residual_normed)
            
            # 保存这层的 centroids (在归一化空间中)
            centroids = kmeans.cluster_centers_.astype(np.float32)
            self.codebooks.append(centroids)     # [codebook_size, embed_dim]
            
            # 计算这层的分配
            assignments = kmeans.labels_    # [num_items,]
            
            # 计算残差(用归一化空间，保证每层尺度一致)
            assigned_centroids = centroids[assignments] # [num_items, embed_dim]
            residual = residual_normed - assigned_centroids
            
            # 打印这层的统计信息
            unique_codes = len(np.unique(assignments))
            inertia = kmeans.inertia_
            print(f" utilization ratio: {unique_codes/self.codebook_size}",
                  f"({100*unique_codes/self.codebook_size :.1f}%)")
            print(f"惯性(inertia): {inertia:.4f}",
                  f"耗时: {time.time() - t0 :.1f}s")
        
        # 训练完成，预计算所有items的SID
        print(f"\n预计算所有 items 的语义ID...")
        self._semantic_ids = self._compute_all_sids(embeddings)
        self._is_fitted = True
        
        # 打印碰撞率
        collision_rate = self._compute_collision_rate()
        print(f"\n训练完成")
        print(f"    碰撞率: {collision_rate:.2f}%")
        return self

    def fit(self, embeddings: np.ndarray) -> 'RKMeans_Tokenizer':
        """与 sklearn 风格一致的训练入口。"""
        return self.forward(embeddings)
    
    def _compute_all_sids(self, embeddings: np.ndarray) -> np.ndarray:
        """
        对所有 item 的 embedding 编码计算, 得到 SID 矩阵
        返回: SID 矩阵 [num_items, num_layers]
        """
        num_items = len(embeddings)
        sids = np.zeros((num_items, self.num_layers), dtype=np.int32)
        residual = embeddings.copy().astype(np.float32)
        
        for layer in range(self.num_layers):
            centroids = self.codebooks[layer]
            
            if self.normalize:
                norms = np.linalg.norm(residual, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8) # 防止除以0
                residual_normed = residual / norms
            else:
                residual_normed = residual
            
            assignments = self._batch_nearest_neighbor(
                residual_normed, centroids
            )
            
            sids[:, layer] = assignments
            
            assigned_centroids = centroids[assignments]
            residual = residual_normed - assigned_centroids
        
        return sids
    
    @staticmethod
    def _batch_nearest_neighbor(queries: np.ndarray,    # [N,D]
                                centroids: np.ndarray,      # [K,D]
                                chunk_size: int = 8192      # 每次处理的item数量，控制内存
                                ) -> np.ndarray:
        """
        批量计算最近邻，分块避免OOM
        
        Returns: [N,]每个query对应的最近centroids index
        """
        N = len(queries)
        assignments = np.zeros(N, dtype=np.int32)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = queries[start:end]  # [chunk, D]
            
            # 计算欧氏距离(利用展开公式避免显式循环)
            # ||q - c||² = ||q||² + ||c||² - 2 q·c^T
            q_norm = (chunk ** 2).sum(axis=1, keepdims=True) # [chunk, 1]
            c_norm = (centroids ** 2).sum(axis=1)[None, :]   # [1, K]
            dot = chunk @ centroids.T   # [chunk, K]
            dists = q_norm + c_norm - 2 * dot
            dists = np.maximum(dists, 0)
            
            assignments[start:end] = np.argmin(dists, axis=1)
            
        return assignments
    
    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """
        单个 embedding -> SID
        Return: SID (c1, c2, c3) [num_layers, ]
        """ 
        assert self._is_fitted, "需要先调用fit()"
        return self._compute_all_sids(embedding[np.newaxis])[0]
    
    def encode_all(self) -> np.ndarray:
        """
        返回训练时预计算的全量 SID

        返回：(num_items, num_layers)
        """
        assert self._is_fitted, "需要先调用 fit()"
        return self._semantic_ids.copy()

    def encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        批量编码新的 embedding（用于推理时的新 item）

        embeddings: (N, embed_dim)
        返回：(N, num_layers)
        """
        assert self._is_fitted, "需要先调用 fit()"
        return self._compute_all_sids(embeddings)
    
    def _compute_collision_rate(self) -> float:
        """计算 SID 碰撞率：有多少 item 共享了相同的 SID"""
        assert self._semantic_ids is not None

        # 把每行 SID 转成字符串，统计唯一值
        sid_strings = [
            tuple(row) for row in self._semantic_ids
        ]
        num_unique  = len(set(sid_strings))
        num_total   = len(sid_strings)
        num_collide = num_total - num_unique

        return 100.0 * num_collide / num_total

    def get_codebook_utilization(self) -> list[float]:
        """
        每层的码本利用率

        返回：List[float]，每层被使用的 code 比例
        """
        assert self._semantic_ids is not None
        utils = []
        for layer in range(self.num_layers):
            unique = len(np.unique(self._semantic_ids[:, layer]))
            utils.append(100.0 * unique / self.codebook_size)
        return utils
                
    def print_stats(self):
        """打印详细统计信息"""
        assert self._is_fitted

        print("=" * 50)
        print("RK-Means 统计信息")
        print("=" * 50)
        print(f"  num_layers:    {self.num_layers}")
        print(f"  codebook_size: {self.codebook_size}")
        print(f"  normalize:     {self.normalize}")
        print(f"  item 总数:     {len(self._semantic_ids)}")
        print()

        # 各层利用率
        utils = self.get_codebook_utilization()
        for layer, util in enumerate(utils):
            print(f"  第{layer+1}层码本利用率: {util:.1f}%")

        # 碰撞率
        collision = self._compute_collision_rate()
        print(f"\n  整体碰撞率: {collision:.2f}%")

        # SID 分布
        sid_counts = {}
        for row in self._semantic_ids:
            key = tuple(row)
            sid_counts[key] = sid_counts.get(key, 0) + 1
        
        max_collision = max(sid_counts.values())
        avg_collision = np.mean(list(sid_counts.values()))
        print(f"  最大碰撞数: {max_collision}（最多有{max_collision}个item共享同一SID）")
        print(f"  平均每个SID对应item数: {avg_collision:.2f}")
        
    def save(self, save_dir: str):
        """
        保存到目录

        save_dir/
          codebooks.npy     所有层的 centroids
          semantic_ids.npy  所有 item 的 SID
          config.pkl        超参数
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存 codebooks
        codebooks_arr = np.stack(self.codebooks)   # (num_layers, K, D)
        np.save(save_dir / "codebooks.npy", codebooks_arr)

        # 保存 SID
        np.save(save_dir / "semantic_ids.npy", self._semantic_ids)

        # 保存超参数
        config = {
            "num_layers":    self.num_layers,
            "codebook_size": self.codebook_size,
            "embed_dim": self.embed_dim,
            "normalize":     self.normalize,
            "batch_size":    self.batch_size,
            "max_iter":      self.max_iter,
            "n_init":        self.n_init,
            "random_state":  self.random_state,
        }
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        print(f"RK-Means 保存到 {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> "RKMeans_Tokenizer":
        """从目录加载"""
        save_dir = Path(save_dir)

        with open(save_dir / "config.pkl", "rb") as f:
            config = pickle.load(f)

        tokenizer = cls(**config)

        codebooks_arr   = np.load(save_dir / "codebooks.npy")
        tokenizer.codebooks = [codebooks_arr[i] for i in range(len(codebooks_arr))]

        tokenizer._semantic_ids = np.load(save_dir / "semantic_ids.npy")
        tokenizer._is_fitted    = True

        print(f"RK-Means 加载自 {save_dir}")
        tokenizer.print_stats()

        return tokenizer