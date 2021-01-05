from typing import List, Tuple

import torch
import torch.optim
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from env import SARSD, Envs
from model import ActorCritic
from reversi import SIZE
import os
import itertools

GAMMA = 0.99
EPISODES = 10_000
SAVE_INTERVAL = 10
NUM_WORKERS = 28
BATCH_SIZE = 32
VALUE_LOSS_COEF = 0.5
ENTROPY_LOSS_CEOF = 0.01

class EpisodeData(Dataset):
    # 为了使用DataLoader
    def __init__(self, data: List[SARSD]):
        super(EpisodeData, self).__init__()
        self.data = []
        # 对数据格式稍微转换一下，并抛弃不需要的数据
        # 因为棋盘具有旋转对称性，对所有状态都进行旋转，增加数据量
        for s, (y, x), R, _, _ in data:
            a = y * SIZE + x
            self.data.append((s, a, R))
            a = SIZE*SIZE - SIZE - SIZE*(a % SIZE) + (a // SIZE) # 推导得出的旋转90度的坐标
            self.data.append((TF.rotate(s, 90), a, R))
            a = SIZE*SIZE - SIZE - SIZE*(a % SIZE) + (a // SIZE)
            self.data.append((TF.rotate(s, 180), a, R))
            a = SIZE*SIZE - SIZE - SIZE*(a % SIZE) + (a // SIZE)
            self.data.append((TF.rotate(s, 270), a, R))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        return self.data[idx]

def main():
    # 确定神经网络计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建神经网络
    net = ActorCritic()
    net = net.to(device)

    # 准备优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    # 准备环境
    envs = Envs(NUM_WORKERS, gamma=GAMMA)
    
    # 开始训练
    for episode in range(EPISODES):

        # 从多个环境采集一回合数据
        net.eval()
        with torch.no_grad():
            states = envs.reset()
            done = False
            while not done:
                states = states.to(device)
                _, policys = net(states)
                policys = policys.cpu() # 移到CPU上处理比较好
                # 不能下的位置概率填 0
                for i in range(NUM_WORKERS):
                    if envs.reversis[i].next != 0:
                        for y, x in itertools.product(range(SIZE), repeat=2):
                            if not envs.reversis[i].good[y][x]:
                                policys[i][y * SIZE + x] = 0.
                            else:
                                policys[i][y * SIZE + x] += 1e-8 # 防止概率全为 0
                actions = Categorical(probs=policys).sample()
                done, states = envs.step(actions)
        
        envs.setReturn()
        data = EpisodeData(envs.readHistory())
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # 训练网络
        net.train()

        # 相关指标
        value_loss_total = 0.
        entropy_total = 0.

        for states, actions, Returns in loader:
            states, actions, Returns = states.to(device), actions.to(device), Returns.to(device)
            values, policys = net(states)

            dist = Categorical(probs=policys)
            action_log_probs = dist.log_prob(actions).view(-1, 1)
            dist_entropy = dist.entropy().mean() # 我们希望分布的熵更大些，保持模型的探索性

            advantages = Returns.view(-1, 1) - values
            
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()

            optimizer.zero_grad()
            (VALUE_LOSS_COEF * value_loss + action_loss - ENTROPY_LOSS_CEOF * dist_entropy).backward()
            optimizer.step()

            value_loss_total += value_loss.item()
            entropy_total += dist_entropy.item()
        
        print('Episode: {:>10d}, Value Loss: {:g}, Entropy: {:g}'.format(
            episode,
            value_loss_total / len(loader),
            entropy_total / len(loader)
            ), flush=True)
        
        if episode != 0 and episode % SAVE_INTERVAL == 0:
            if not os.path.isdir('models'):
                os.mkdir('models')
            torch.save(net.state_dict(), 'models/{}.pt'.format(episode // SAVE_INTERVAL))

if __name__ == '__main__':
    main()