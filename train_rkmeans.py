import numpy as np
import pickle
from models.RKmeans import RKMeans_Tokenizer

CONFIG = {
    "embedding_path": "datasets/processed/item_embeddings.npy",
    "data_path":      "datasets/processed/beauty.pkl",
    "save_dir":       "checkpoints/rkmeans/",
    "sid_save_path":  "datasets/processed/semantic_ids_rkmeans.npy",

    # RK-Means 超参数
    "num_layers":    4,
    "codebook_size": 256,
    "normalize":     False,    # 论文建议：RK-Means 需要归一化 --> 这里不行
    "batch_size":    4096,
    "max_iter":      1000,
    "n_init":        3,
    "random_state":  42,
}


def train_rkmeans(config=CONFIG):

    # 加载 embedding 
    print(f"加载 item embeddings: {config['embedding_path']}")
    embeddings = np.load(config["embedding_path"])
    print(f"  embedding shape: {embeddings.shape}")

    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)
    print(f"  item 总数: {data['num_items']}")

    # 确认维度一致
    assert len(embeddings) == data["num_items"], (
        f"embedding 数量 {len(embeddings)} 和 "
        f"item 数量 {data['num_items']} 不一致"
    )

    # 训练 
    tokenizer = RKMeans_Tokenizer(
        num_layers=config["num_layers"],
        codebook_size=config["codebook_size"],
        embed_dim=embeddings.shape[1],
        normalize=config["normalize"],
        batch_size=config["batch_size"],
        max_iter=config["max_iter"],
        n_init=config["n_init"],
        random_state=config["random_state"],
    )
    
    tokenizer.fit(embeddings)
    
    tokenizer.save(config['save_dir'])
    
    # 把 SID 单独保存到 data/processed/，方便推荐模型直接加载
    semantic_ids = tokenizer.encode_all()
    np.save(config["sid_save_path"], semantic_ids)
    print(f"\n语义ID保存到 {config['sid_save_path']}")
    print(f"  shape: {semantic_ids.shape}")
    
    tokenizer.print_stats()
    
    return tokenizer


if __name__ == "__main__":
    train_rkmeans()