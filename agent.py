from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from reversi import SIZE

class Block(nn.Module): # 残差块
    def __init__(self, in_planes: int, out_planes: int, stride: int, group_planes: int,
        conv_shortcut: bool):

        super(Block, self).__init__()
        hidden_planes = out_planes
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        # 3x3卷积核
        groups = hidden_planes // group_planes
        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_planes)
        # 1x1卷积核
        self.conv3 = nn.Conv2d(hidden_planes, out_planes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        if conv_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out

class Policy(nn.Module): # 使用残差CNN结构的Actor-Critic网络
    def __init__(self, in_channels: int):
        super(Policy, self).__init__()
        self.size = SIZE
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.last_plane = 64

        self.depths = [1, 1, 4, 7]
        self.planes = [24, 56, 152, 368]
        self.strides = [1, 1, 2, 2]
        self.group_planes = 8

        self.layer1 = self.make_layer(0)
        self.layer2 = self.make_layer(1)
        self.layer3 = self.make_layer(2)
        self.layer4 = self.make_layer(3)

        self.critic_linear = nn.Linear(self.planes[-1], 1)
        self.actor_linear = nn.Linear(self.planes[-1], SIZE*SIZE)
    
    def make_layer(self, idx: int) -> nn.Module:
        depth = self.depths[idx]
        plane = self.planes[idx]
        stride = self.strides[idx]
        group_planes = self.group_planes

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(
                Block(self.last_plane, plane, s, group_planes, i == 0)
            )
            self.last_plane = plane
        
        return nn.Sequential(*layers)
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn1(self.conv1(states)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        value = self.critic_linear(out)
        policy = self.actor_linear(out)
        return value, policy
    
# def A2CAgent:
#     def __init__(self):
#         self.