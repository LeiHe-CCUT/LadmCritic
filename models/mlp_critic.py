import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Critic(nn.Module):
    """
    一個標準的、通用的多層感知機 (MLP) Critic 網路。
    它遵循 SAC 的 Twin Q-Network 架構，包含兩個獨立的 Q 網路 (q1, q2)。
    """
    def __init__(self, state_dim, action_dim):
        super(MLP_Critic, self).__init__()

        # --- Q 網路 1 ---
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # --- Q 網路 2 ---
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        前向傳播，計算兩個 Q 值。
        """
        # 將狀態和動作拼接在一起作為輸入
        sa = torch.cat([state, action], 1)

        # 計算 Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # 計算 Q2
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
