from typing import List, Tuple, Deque

import torch
from reversi import Coordinate, Reversi

# state, action, reward, next_state, done
SARSD = Tuple[torch.Tensor, Coordinate, int, torch.Tensor, bool]

class 名字:
    def __init__(self, num_workers: int, replay: Deque[SARSD]): # ...
        pass

    # state格式为(3, SIZE, SIZE)，全都是0或1，第一张图是黑棋位置，第二张图是白棋位置（1表示有）
    # 最后一张全0表示下一步是白棋走全1表示是黑棋走

    # reward设计为：如果当前棋子比对方多，奖励为1，否则为-1，胜利奖励为100

    def reset(self) -> torch.Tensor: # (num_worker, 3, SIZE, SIZE)
        pass

    def step(actions: List[Coordinate]) -> Tuple[bool, torch.Tenosr]:
        # 返回值：第一个值表示所有环境是否结束，第二个是next_state (num_worker, 3, SIZE, SIZE)

        # 注意每一步都要append进replay！
        pass
