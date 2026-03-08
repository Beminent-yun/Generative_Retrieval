import pandas as pd
from tqdm.auto import tqdm
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

# 优先使用 orjson（快 3~5 倍），若未安装则回退到标准 json
try:
    import orjson as json_lib
    _loads = json_lib.loads
except ImportError:
    import json as json_lib
    _loads = json_lib.loads


def _iter_jsonl_gz(path: Path):
    """逐行生成解析后的 dict，流式读取不在内存中堆积原始记录"""
    with gzip.open(path, "rb") as f:           # 二进制模式，orjson/json 均可接受 bytes
        for line in tqdm(f, desc=f"解析 {path.name}"):
            line = line.strip()
            if not line:
                continue
            try:
                yield _loads(line)
            except Exception:
                pass                            # 彻底损坏的行直接丢弃


def parse_jsonl_gz(path: Path) -> list:
    """
    解析 gzip 压缩的 jsonl 文件（保留原接口供外部调用）
    每行是一个独立的 JSON 对象
    """
    path = Path(path)
    return list(_iter_jsonl_gz(path))


def load_reviews(path: Path) -> pd.DataFrame:
    """
    加载交互数据，只保留需要的四个字段：(user, item, rating, timestamp)
    优化：流式解析 + dict-of-lists 构建 DataFrame，比 list-of-dicts 快约 2 倍
    return: DataFrame
    """
    path = Path(path)

    # 用 dict-of-lists 直接收集字段，避免中间对象开销
    # 兼容旧版(2018: reviewerID/overall/unixReviewTime)与新版(2023: user_id/rating/timestamp)
    users, items, ratings, timestamps = [], [], [], []
    for r in _iter_jsonl_gz(path):
        uid = r.get("user_id") or r.get("reviewerID", "")
        iid = r.get("parent_asin") or r.get("asin", "")
        ts_raw = r.get("timestamp") or r.get("unixReviewTime", 0)
        rating_raw = r.get("rating") or r.get("overall", 0)

        try:
            ts = int(ts_raw)
            # 新版时间戳是毫秒，转为秒
            if ts > 1e12:
                ts = ts // 1000
        except (ValueError, TypeError):
            ts = 0

        if not uid or not iid or ts <= 0:
            continue
        users.append(uid)
        items.append(iid)
        ratings.append(float(rating_raw))
        timestamps.append(ts)

    df = pd.DataFrame({"user": users, "item": items, "rating": ratings, "timestamp": timestamps})
    df = df.drop_duplicates(["user", "item"])   # 同一用户对同一商品只保留一条交互

    print(f"加载交互: {len(df)}条 | {df['user'].nunique()} 用户 {df['item'].nunique()} 物品")
    return df


def load_meta(path: Path) -> Dict[str, dict]:
    """
    加载商品元数据
    Return: {asin: {title, description, category, brand}}
    """
    path = Path(path)
    meta = {}
    for r in _iter_jsonl_gz(path):
        # 新版用 parent_asin，旧版用 asin
        asin = r.get('parent_asin') or r.get('asin', '')
        if not asin:
            continue
        title = r.get('title', '')
        desc_raw = r.get('description', "")
        if isinstance(desc_raw, list):
            description = " ".join(desc_raw)
        else:
            description = str(desc_raw)

        cats = r.get("categories", [])
        if cats:
            if isinstance(cats[0], list):
                # 旧版：嵌套列表 [["A", "B"]]
                category = " > ".join(cats[0])
            else:
                # 新版：平铺字符串列表 ["A", "B"]
                category = " > ".join(cats)
        else:
            category = ""

        # 新版 brand 可能在顶层，也可能在 details 字典中
        brand = r.get('brand') or r.get('details', {}).get('Brand', "") or ""

        meta[asin] = {
            'title': title,
            'description': description,
            'category': category,
            'brand': brand
        }
        
    print(f"加载元数据: {len(meta)} 个商品")
    return meta


def kcore_filter(df:pd.DataFrame, k:int=5) -> pd.DataFrame:
    """
    K-core 过滤： 反复过滤，直到所有用户和商品都至少有k条交互
    为什么需要反复过滤：
      第一轮过滤掉用户A（交互太少）
      导致某些商品的交互数减少
      第二轮可能需要再过滤那些商品
      如此反复，直到稳定
    """
    before = len(df)
    iteration = 0
    
    while True:
        iteration += 1
        
        # 过滤交互少于k次的用户
        user_counts = df['user'].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df['user'].isin(valid_users)]
        
        # 过滤交互少于k次的商品
        item_counts= df['item'].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df['item'].isin(valid_items)]
        
        after = len(df)
        print(f"  第{iteration}轮: {after:,} 条 "
              f"({df['user'].nunique():,} 用户, {df['item'].nunique():,} 商品)")
        
        # 稳定了就停止
        if after == before:
            break
        before = after
    
    return df.reset_index(drop=True)


def filter_by_rating(df:pd.DataFrame, min_rating: float=4.0) -> pd.DataFrame:
    """
    只保留高评分交互
    
    为什么这样做：
      评分1-2的交互是负面信号，不能算作"用户喜欢这个商品"
      推荐系统通常只建模正向交互
      4分以上视为"隐式正反馈"
    
    注意：有些论文不做这个过滤，认为所有交互（包括差评）都是信号
    可以根据实验效果决定是否保留
    """
    before = len(df)
    df = df[df['rating'] >= min_rating]
    print(f"评分过滤（>={min_rating}）: {before:,} → {len(df):,} 条")
    return df


def build_id_maps(df: pd.DataFrame) -> Tuple[dict, dict, dict, dict]:
    """
    把字符串 ID 映射到连续整数 (即将字符串转换成数值型给计算机看)
    Returns:
        - user2id
        - item2id
        - id2user
        - id2item
    """
    # 排序保证结果运行一致
    all_users = sorted(df['user'].unique().tolist())
    all_items = sorted(df['item'].unique().tolist())
    
    user2id = {u: i for i, u in enumerate(all_users)}
    id2user = {i: u for i, u in enumerate(all_users)}
    item2id = {it: i for i, it in enumerate(all_items)}
    id2item = {i: it for i, it in enumerate(all_items)}
    
    return user2id, item2id, id2user, id2item


def build_behavior_seq(
    ratings_df: pd.DataFrame,
    user2id: dict,
    item2id: dict,
    max_len: int = 50
) -> Dict[int, List[int]]:
    """
    为每个用户构建按时间排序的交互序列
    - 为什么按时间排序：
      序列推荐的核心假设是用户行为有时序性
      "先买了A，再买了B"和"先买了B，再买了A"代表不同的偏好演化
    
    - 为什么截断到 max_len：
      Transformer 的计算复杂度是 O(T²)，序列太长训练很慢
      实际上距离当前时间太远的历史对预测的贡献很小
      截断最近 max_len 个交互
    
    Returns: {user_id: [item_id#1, ...]} 按时间从旧到新
    """
    df = ratings_df.copy()
    df['user'] = df['user'].map(user2id)
    df['item'] = df['item'].map(item2id)
    
    # 按时间排序
    df = df.sort_values(['user', 'timestamp'])
    
    # 构建序列
    sequences = {}
    for user_id, group in df.groupby('user'):   # group是DataFrame子集，user_id对应的所有行
        items = group['item'].tolist()
        sequences[user_id] = items[-max_len:]   # 截断至max_len，即[0:max_len-1]
    
    # 统计序列长度分布
    lengths = [len(s) for s in sequences.values()]
    print(f"序列长度：min = {min(lengths)}",
          f"max = {max(lengths)}",
          f"mean = {sum(lengths)/len(lengths)}")
    
    return sequences


def leave_one_out_split(
    sequences: Dict[int, List[int]],
    min_seq_len: int = 3
)->Tuple[dict, dict, dict]:
    """
    Leave-One-Out 划分
    为什么使用 leave-one-out:
        保证每个用户都有 train/val/test 数据
        测试集用真实的“下一次交互”，比随机划分更真实
    
    划分规则：
        test: 最后1个item(最新交互) -> 用最后1个而不是最后几个，模拟“预测用户下一次会买什么”。一次只预测一个，和实际推荐场景一致
        val: 倒数第2个item
        train: 剩余所有item(历史)
    
    min_seq_len = 3: 保证至少有1个历史数据/train + 1val + 1test
    """
    train, val, test = {}, {}, {}
    skipped = 0
    
    for user_id, seq in sequences.items():
        if len(seq) < 3:
            skipped += 1
            continue
        train[user_id] = seq[:-2]   # 历史序列(可能很长)
        val[user_id] = seq[-2]      # 整数
        test[user_id] = seq[-1]     # 整数
    
    print(f"Leave-one-out 划分:")
    print(f"  Train: {len(train):,} 用户")
    print(f"  Val:   {len(val):,} 用户")
    print(f"  Test:  {len(test):,} 用户")
    print(f"  跳过（序列太短）: {skipped} 用户")
    
    return train, val, test

import re
# 特征工程：构建item文本
def clean_text(text:str, max_len:int = 256) -> str:
    """
    清洗文本
    处理顺序：
    - 去掉HTML标签（元数据里有大量<br>, <p>等）
    - 合并多余空格
    - 截断(控制在[0,max_len-1]):防止Sentence-BERT处理超长文本
    """
    if not text:
        return ""
    # 去掉HTML标签
    text = re.sub(r"<[^>]+>", " ", text)
    # 合并空白
    text = re.sub(r"\s+", " ", text).strip()
    # 截断
    return text[:max_len]

def build_item_texts(
    all_items_asin: List[str],  # 按item_id 排序的asin列表
    meta: Dict[str, Dict]
)->Tuple[List[str], List[str]]:
    """
    为每个item构建用于Sentence-BERT的文本表示
    文本格式设计：
        "Title: {title}. Brand: {brand}. Category: {category}. {description}"
    - 为什么这样设计：
      Sentence-BERT 对格式化的输入效果更好
      显式标注字段名（Title:, Brand:）帮助模型区分不同属性
      Title 排在最前面，因为它信息密度最高
      Description 排最后，因为可能很长，截断影响最小
      
    Returns:
        -item_texts: List[str], 用于提取embedding
        -item_titles: List[str], 用于展示和调试
    """
    item_texts = []
    item_titles = []
    missing = 0
    
    for asin in all_items_asin:
        m = meta.get(asin, {})
        
        if not m:
            # 没有元数据的商品，用asin本身作为文本
            # 至少有一个非空字符串，否则sentence-bert报错
            item_texts.append(f"Product {asin}")
            item_titles.append(asin)
            missing += 1
            continue
        
        title = clean_text(m.get('title', ''), max_len=128)
        brand = clean_text(m.get('brand', ''), max_len=64)
        category = clean_text(m.get('category', ''), max_len=128)
        description = clean_text(m.get('description', ''), max_len=256)
        
        # 拼接各字段，跳过空字段
        parts = []
        if title: parts.append(f"Title {title}")
        if brand: parts.append(f"Brand {brand}")
        if category: parts.append(f"Category {category}")
        if description: parts.append(description)
        
        text = ". ".join(parts) if parts else f"Product {asin}"
        
        item_texts.append(text)
        item_titles.append(title or asin)
    
    coverage = (len(all_items_asin) - missing) / len(all_items_asin)
    print(f"元数据覆盖率: {coverage:.1%} "
          f"({missing} 个商品无元数据)")
    
    return item_texts, item_titles

import pickle

def main(
    review_path: Path,
    meta_path: Path,
    output_path: Path,
    min_rating:int = 4,
    kcore:int = 5,
    max_seq_len: int = 50
):
    review_path = Path(review_path)
    meta_path = Path(meta_path)
    output_path = Path(output_path)

    df = load_reviews(review_path)
    meta = load_meta(meta_path)
    
    df = kcore_filter(df, kcore)
    df = filter_by_rating(df, min_rating) # 获取正样本评分的数据集
    
    user2id, item2id, id2user, id2item = build_id_maps(df)
    
    all_items_asin = [id2item[i] for i in range(len(id2item))]  # 按照item_id将item_asin排序好
    
    sequences = build_behavior_seq(df, user2id, item2id, max_seq_len)
    
    train, val, test = leave_one_out_split(sequences)
    
    item_texts, item_titles = build_item_texts(all_items_asin, meta)
    
    data = {
        # ID mapping
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
        
        # Sequence Split
        "train": train,
        "val": val,
        "test": test,
        
        # item feature
        "item_texts": item_texts,
        "item_titles": item_titles,
        
        # Statistic
        "num_users": len(user2id),
        "num_items": len(item2id)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"保存到: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return data


if __name__ == "__main__":
    review_path = "datasets/Beauty_and_Personal_Care.jsonl.gz"
    meta_path = "datasets/meta_Beauty_and_Personal_Care.jsonl.gz"
    output_path = "datasets/processed/beauty.pkl"
    main(review_path,
         meta_path,
         output_path)
    