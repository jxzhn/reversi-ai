import torch
from torch.distributions.categorical import Categorical
from reversi import Coordinate, Reversi, SIZE
from model import ActorCritic
from env import getBoardState
import itertools

class Agent:
    def __init__(self):
        self.net = ActorCritic()
        self.net.load_state_dict(torch.load('models/good.pt', map_location='cpu'))
        self.net.eval()
        torch.no_grad() # 关闭梯度记录
    
    def brain(self, reversi: Reversi, who: int) -> Coordinate:
        # assert reversi.next == who
        state = torch.Tensor(getBoardState(reversi)).unsqueeze(0)
        policy = self.net(state)[1][0]

        # 保证位置合法性
        for y, x in itertools.product(range(SIZE), repeat=2):
            if not reversi.good[y][x]:
                policy[y * SIZE + x] = 0.
            else:
                policy[y * SIZE + x] += 1e-8 # 防止概率全为 0
        
        action = policy.max(dim=-1).indices.item()
        return (action // SIZE, action % SIZE)
        