from typing import Iterable, List, Tuple

from multiprocessing import Pool
import torch
import itertools
from reversi import Coordinate, Reversi, SIZE

# state, action, Return, next_state, done
SARSD = Tuple[torch.Tensor, Coordinate, float, torch.Tensor, bool]
Matrix = List[List[int]]

# state格式为(3, SIZE, SIZE)，全都是0或1，第一张图是黑棋位置，第二张图是白棋位置（1表示有）
# 最后一张全0表示下一步是白棋走全1表示是黑棋走
# reward设计为：如果当前棋子比对方多，奖励为1，否则为-1，胜利奖励为100，失败奖励为-100

# 辅助函数，获得当前棋盘的Tensor，输入：棋盘编号；输出：3*SIZE*SIZE的List
def getBoardState(reversi: Reversi) -> List[Matrix]:
    next = 2 - reversi.next # 等价于 1 if reversi.next == 1 else 0 因为 self.next 为 1 或 2
    return [
        [[int(reversi.board[y][x] == 1) for x in range(SIZE)] for y in range(SIZE)], # 黑棋位置
        [[int(reversi.board[y][x] == 2) for x in range(SIZE)] for y in range(SIZE)], # 白棋位置
        [[next for _ in range(SIZE)] for _ in range(SIZE)]
    ]

# 辅助函数，检查place返回的状态，返回一个int，-1：对局未结束，0：平局，1：黑赢，2：白赢
def checkPlaceStatus(status: str) -> int:
    if status.startswith('end'):
        return int(status[-1])
    elif status == 'ok':
        return -1
    elif status != 'ok':
        raise Exception('Should not reach here!')

# 辅助函数，让当前棋局走一步，返回新棋盘、是否结束以及五元组
def takeAction(arg: Tuple[Reversi, Coordinate, bool]) -> Tuple[Reversi, bool, SARSD]:
    reversi, action, end = arg
    # 棋局已结束
    if end:
        return reversi, True, None

    # 保存走之前的状态
    s = torch.Tensor(getBoardState(reversi))
    
    # 走棋
    status = reversi.place(action, reversi.next)
    result = checkPlaceStatus(status)
    end = (result != -1)

    # 获取更新的状态
    sn = torch.Tensor(getBoardState(reversi))
    
    # 计算reward，暂存在Return里
    a = reversi.next        # 己方
    b = 2 if a == 1 else 1  # 敌方
    
    if result == -1 and reversi.number[a] > reversi.number[b]:
        reward = 1
    elif result == -1 and reversi.number[a] < reversi.number[b]:
        reward = -1
    elif result == a:
        reward = 100
    elif result == b:
        reward = -100
    else:
        reward = 0

    return reversi, end, [s, action, reward, sn, end]


class Envs:
    def __init__(self, num_workers: int, gamma: float):
        self.num_workers = num_workers
        self.gamma = gamma
        self.historys = [[] for i in range(num_workers)] # 分开记录每个棋局的历史记录
        self.end = [False for i in range(num_workers)] # 某个worker的棋局是否结束
        self.reversis = [Reversi() for i in range(num_workers)]
        self.pool = Pool(self.num_workers)

    # 重置所有的Reversi实例
    def reset(self) -> torch.Tensor: # (num_worker, 3, SIZE, SIZE)
        self.historys = [[] for _ in range(self.num_workers)]
        self.reversis = [Reversi() for _ in range(self.num_workers)]
        self.end = [False for _ in range(self.num_workers)]
        states = self.pool.map(getBoardState, self.reversis)
        return torch.Tensor(states)

    # 让所有棋盘都走一步，输入：坐标列表
    # 返回值：第一个值表示所有环境是否结束，第二个是next_state (num_worker, 3, SIZE, SIZE)
    def step(self, actions_in_int: Iterable[int]) -> Tuple[bool, torch.Tensor]:
        # 获取action的坐标表示
        actions = [(a // SIZE, a % SIZE) for a in actions_in_int]
        
        # 并行执行环境交互
        retvals = self.pool.map(takeAction, zip(self.reversis, actions, self.end))

        self.reversis = [retval[0] for retval in retvals]
        self.end = [retval[1] for retval in retvals]
        infos = [retval[2] for retval in retvals]

        # 附加到history中
        for i in range(self.num_workers):
            if infos[i] != None:
                self.historys[i].append(infos[i])
        
        return sum(self.end) == self.num_workers, torch.stack([info[3] if info else torch.zeros((3, SIZE, SIZE)) for info in infos])

    # 从后向前更新Return（这里我懒得搞并行了，感觉速度差别不大）
    def setReturn(self):
        for history in self.historys:
            R = 0
            for t in range(len(history)-1 , -1, -1):
                R = history[t][2] + self.gamma * R
                history[t][2] = R

    # 取得合并后的history
    def readHistory(self) -> List[SARSD]:
        history_all = []
        for history in self.historys:
            history_all.extend(history)
        return history_all

# 顺序：
# 1：m.reset()
# 2：每次要走下一步 m.step([44, 44, 44, 44])
# 3：所有棋局都结束之后 m.setReturn()
# 4：读取路径记录 h = m.readHistory()
if __name__ == '__main__':
    e = Envs(2, gamma=0.99)
    import random
    def randomAgent(reversi: Reversi):
        available = [(y, x) for (y, x) in itertools.product(range(reversi.size), repeat=2) if reversi.good[y][x]]
        if not available:
            print('No available position')
            return (-1, -1)
        return random.choice(available)
    
    import time
    begin = time.time()

    for i in range(100):
        print(i, end='\r')
        e.reset()
        idx = 0
        while True:
            y1, x1 = randomAgent(e.reversis[0])
            y2, x2 = randomAgent(e.reversis[1])    
            actions = [y1 * SIZE + x1, y2 * SIZE + x2]
            finish, _ = e.step(actions)
            if finish:
                break
        e.setReturn()
        h = e.readHistory()
    
    print('{:g}s took to finish 100 episodes.'.format(time.time() - begin))