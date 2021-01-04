from typing import List, Tuple, Deque
from multiprocessing import Pool
import torch
import itertools
from reversi import Coordinate, Reversi, SIZE
gamma = 1.0 # 超参数，为了测试这里先用变量代替

# state, action, Return, next_state, done
SARSD = Tuple[torch.Tensor, Coordinate, float, torch.Tensor, bool]

class multiplayer:
    def __init__(self, num_workers: int, gamma: float): # ...
        self.num_workers = num_workers
        self.gamma = gamma
        self.historys = [[] for i in range(num_workers)] # 分开记录每个棋局的历史记录
        self.end = [False for i in range(num_workers)] # 某个worker的棋局是否结束
        self.reversis = [Reversi() for i in range(num_workers)]

    # state格式为(3, SIZE, SIZE)，全都是0或1，第一张图是黑棋位置，第二张图是白棋位置（1表示有）
    # 最后一张全0表示下一步是白棋走全1表示是黑棋走
    # reward设计为：如果当前棋子比对方多，奖励为1，否则为-1，胜利奖励为100，失败奖励为-100

    # 重置所有的Reversi实例
    def reset(self) -> torch.Tensor: # (num_worker, 3, SIZE, SIZE)
        for i in range(self.num_workers):
            self.historys[i].clear() # 重置是清空棋盘记录
        pool = Pool(self.num_workers)
        args = [(self.reversis, idx) for idx in range(self.num_workers)]
        info = pool.map(resetABoard, args)
        info = torch.Tensor(info)
        pool.close()
        pool.join()
        # print(info.size())
        return info

    # 让所有棋盘都走一步，输入：坐标列表
    # 返回值：第一个值表示所有环境是否结束，第二个是next_state (num_worker, 3, SIZE, SIZE)
    def step(self, actions_in_int: List[int]) -> Tuple[List[bool], torch.Tensor]:
        # 获取action的坐标表示
        actions = []
        for i in range(self.num_workers):
            y = actions_in_int[i] // SIZE
            x = actions_in_int[i] - (y * SIZE)
            actions.append((y, x))
        # 并行生成history
        pool = Pool(self.num_workers)
        args = [(self.reversis, actions, self.end, idx) for idx in range(self.num_workers)]
        info = pool.map(getHistory, args)
        pool.close()
        pool.join()
        # 附加到history中
        for idx in range(self.num_workers):
            if info[idx] != None:
                self.historys[idx].append(info[idx])
        
        # 更新棋盘（如果不更新，上面的结果不能同步到这个类的数据成员里，idk why，py没学好……）
        for i in range(self.num_workers):
            if not self.end[i]:
                a = self.reversis[i].next # 记录这一步是谁走的
                b = 2 if a == 1 else 1 
                result = -1
                if self.reversis[i].next == 1: # 这一步走黑棋
                    status = self.reversis[i].place(actions[i], 1)
                    result = checkPlaceStatus(status)
                elif self.reversis[i].next == 2: # 走白棋
                    status = self.reversis[i].place(actions[i], 2)
                    result = checkPlaceStatus(status)
                if result != -1:
                    end = True

        finish = (sum(self.end) == self.num_workers)
        for idx in range(self.num_workers):
            if info[idx] == None:
                info[idx][3] = [[[0 for i in range(SIZE)] for i in range(SIZE)] for i in range(3)]
        next_state = [info[idx][3] for idx in range(self.num_workers)]
        next_state = torch.tensor(next_state)
        return finish, next_state

    # 从后向前更新Return（这里我懒得搞并行了，感觉速度差别不大）
    def setReturn(self):
        for i in range(self.num_workers):
            R = 0
            for t in range( len(self.historys[i])-1 , -1, -1):
                R = self.historys[i][t][2] + self.gamma * R # reward[t] + gamma * R
                self.historys[i][t][2] = R # s[t].R = R

    # 取得合并后的history
    def getHistory(self) -> List[SARSD]:
        history = []
        for i in range(self.num_workers):
            for j in range(len(self.historys[i])):
                s = torch.tensor(self.historys[i][j][0])
                sn = torch.tensor(self.historys[i][j][3])
                sarsd = (s, self.historys[i][j][1], self.historys[i][j][2], sn, self.historys[i][j][4]) # 处理成Tuple的形式
                history.append(sarsd)
        return history

# 辅助函数，重置指定棋盘
def resetABoard(arg) -> List:
    idx = arg[1]
    reversi = arg[0][idx]
    reversi = Reversi()
    board = getBoard(reversi)
    return board

# 辅助函数，获得当前棋盘的Tensor，输入：棋盘编号；输出：棋盘信息列表3*SIZE*SIZE
def getBoard(reversi) -> List:
    black = [[0 for i in range(SIZE)] for i in range(SIZE)]
    white = [[0 for i in range(SIZE)] for i in range(SIZE)]
    next_player = [[0 for i in range(SIZE)] for i in range(SIZE)] # 初始化的时候默认下一步是白棋
    # 1：黑棋；2：白棋
    for (y, x) in itertools.product(range(SIZE), repeat=2):
        if reversi.board[y][x] == 1:
            black[y][x] = 1
        if reversi.board[y][x] == 2:
            white[y][x] = 1
    return [black, white, next_player]

# 辅助函数，让当前棋局走一步，并且返回棋盘这一步的SARSD
def getHistory(arg):
    idx = arg[3]
    reversi = arg[0][idx]
    action = arg[1][idx]
    end = arg[2][idx]

    # 如果棋局没结束，就继续更新history
    if not end:
        # 保存走之前的状态
        s = getBoard(reversi)
        
        # 走棋
        a = reversi.next # 记录这一步是谁走的
        b = 2 if a == 1 else 1 
        if reversi.next == 1: # 这一步走黑棋
            status = reversi.place(action, 1)
            result = checkPlaceStatus(status)
            if result != -1:
                end = True
        elif reversi.next == 2:
            status = reversi.place(action, 2)
            result = checkPlaceStatus(status)
            if result != -1:
                end = True
        # 获取更新的状态
        sn = getBoard(reversi)
        
        # 计算reward，暂存在Return里
        reward = 0
        if result == -1 and reversi.number[a] > reversi.number[b]:
            reward = 1
        elif result == -1 and reversi.number[a] < reversi.number[b]:
            reward = -1
        elif result != 0 and result == a:
            reward = 100
        elif result != 0 and result == b:
            reward = -100
        
        # 把SARAD返回，在step中更新
        return [s, action, reward, sn, end]
    else: 
        return None # 这里为了鲁棒性，已经结束的棋局返回None

# 辅助函数，检查place返回的状态，返回一个int：-1：对局未结束；0：平局；1：黑赢：2：白赢
def checkPlaceStatus(status: str) -> int:
    if status.startswith('end'):
        return int(status[-1])
    elif status == 'ok':
        return -1
    elif status != 'ok':
        raise Exception('Should not reach here!')

# test
m = multiplayer(4, gamma)
m.reset()
m.step([44, 44, 44, 44])
m.setReturn()
h = m.getHistory()
m.step([29, 29, 29, 29])
m.setReturn()
h = m.getHistory()
