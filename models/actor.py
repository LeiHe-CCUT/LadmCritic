# # models/actor.py
# import torch
# import torch.nn as nn
# from torch.distributions import Normal

# LOG_STD_MAX = 2
# LOG_STD_MIN = -20

# class Actor(nn.Module):
#     """ A standard Gaussian policy Actor network. """
#     def __init__(self, state_dim, action_dim, max_action):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#         )
#         self.mean_layer = nn.Linear(256, action_dim)
#         self.log_std_layer = nn.Linear(256, action_dim)
        
#         self.max_action = max_action

#     def forward(self, state):
#         x = self.net(state)
#         mean = self.mean_layer(x)
        
#         log_std = self.log_std_layer(x)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)
        
#         return mean, std

#     def sample(self, state):
#         mean, std = self.forward(state)
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick
#         y_t = torch.tanh(x_t)
#         action = y_t * self.max_action
        
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)
        
#         return action, log_prob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    """ 
    优化后的高可解释性 Actor 网络
    集成了 CWN (上下文加权网络) 分支，支持逻辑变量提取
    """
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        
        # 1. 共享骨干网络 (Shared Backbone)
        self.base_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # 2. 动作分支 (Action Heads)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        
        # 3. [新增] CWN 分支：计算上下文权重 ws, we, wc
        self.cwn_layer = nn.Linear(256, 3) 
        
        # 4. [新增] 风险分支：计算物理风险势能 U_risk
        self.risk_layer = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, state):
        """ 前向传播，返回所有决策分量 """
        x = self.base_net(state)
        
        # 计算动作分布参数
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # [逻辑提取] 计算 CWN 权重 (使用 Softmax 确保三者之和为 1)
        # weights 包含 [ws, we, wc]
        weights = torch.softmax(self.cwn_layer(x), dim=-1)
        
        # [逻辑提取] 计算物理风险能 U (使用 Softplus 确保风险为正值)
        u_risk = F.softplus(self.risk_layer(x))
        
        return mean, std, weights, u_risk

    def sample(self, state, return_logic=False):
        """ 
        采样动作并支持逻辑变量返回
        :param return_logic: 是否返回内部权重和风险能，用于图 6 绘制
        """
        # 调用 forward 获取所有输出
        mean, std, weights, u_risk = self.forward(state)
        
        # 标准 SAC 采样逻辑 (Reparameterization Trick)
        normal = Normal(mean, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # 计算 Log Probability (用于 SAC 更新)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 如果开启了逻辑返回模式 (专门针对图 6 的采集脚本)
        if return_logic:
            # 将 Tensor 转换为可记录的数值字典
            # 注意：这里我们提取 batch 中的第一个样本 (.item())
            logic_vars = {
                'ws': weights[0, 0].item(), # 安全权重
                'we': weights[0, 1].item(), # 效率权重
                'wc': weights[0, 2].item(), # 舒适度权重
                'u_risk': u_risk[0, 0].item() # 物理势能
            }
            return action, log_prob, logic_vars
        
        return action, log_prob

    def get_action_only(self, state):
        """ 极速推理接口，仅用于部署 """
        mean, _ , _, _ = self.forward(state)
        return torch.tanh(mean) * self.max_action