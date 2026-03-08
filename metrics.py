from typing import Dict, List, Tuple
import numpy as np


def hr_at_k(recommended: List[int], target:int, k:int) -> float:
    """
    HR@K: Hit Rate, 目标item是否出现在推荐列表的前K位
    return:
        1.0 -> 命中
        0.0 -> 未命中
    """
    return float(target in recommended[:k])

def ndcg_at_k(recommended: List[int], target:int, k:int) -> float:
    """
    NDCG@K: 考虑命中的折损收益
    
    target 命中位置越靠前，分数越高：
        rank=1 -> 1/log2(2) = 1.000
        rank=2 -> 1/log2(3) = 0.631
        rank=3 -> 1/log2(4) = 0.500
        rank=k -> 1/log2(k+1)
        未命中 -> 0.0
    """
    if target not in recommended[:k]:
        return 0.0
    rank = recommended.index(target) + 1
    return 1/np.log2(rank + 1)


