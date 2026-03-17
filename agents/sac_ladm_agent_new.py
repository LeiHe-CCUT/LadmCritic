import torch
import torch.nn.functional as F
import numpy as np
import os
from models.actor import Actor
from models.ladm_critic import LadmCritic

class SAC_Ladm_Agent:
    def __init__(self, state_dim, action_dim, max_action, config):
        # 1. 设备初始化：优先遵循 config 配置
        self.device = torch.device(config.get('device', "cuda" if torch.cuda.is_available() else "cpu"))
        
        # 2. 创建 Actor 网络 (Policy)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

        # 3. 创建 LadmCritic 网络 (神级裁判)
        # 注意：这里的 Critic 内部包含了物理图网络 (PA-GAT)
        self.critic = LadmCritic(state_dim, action_dim, config).to(self.device)
        self.critic_target = LadmCritic(state_dim, action_dim, config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        
        # 4. 超参数
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.max_action = max_action
        
        # [新增] 专门用于定性分析记录的变量
        self.logic_record = {}

    def select_action(self, state, evaluate=False, return_logic=False):
        """
        选择动作。
        :param state: 原始观测状态
        :param evaluate: 是否为评估模式（不增加噪声）
        :param return_logic: 开启后返回神经网络内部的权重和物理能，用于画 4x3 矩阵图
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        if evaluate:
            # 评估模式：取均值动作
            action, _, logic_vars = self.actor.sample(state, return_logic=True)
        else:
            # 训练模式：采样带噪声的动作
            action, _, logic_vars = self.actor.sample(state, return_logic=True)

        action_np = action.cpu().data.numpy().flatten()

        if return_logic:
            # 提取 Actor 内部输出的物理逻辑变量
            # 假设 Actor.sample 返回的 logic_vars 结构为: {'ws': ..., 'we': ..., 'wc': ..., 'u_risk': ...}
            return action_np, logic_vars
        
        return action_np

    def update(self, replay_buffer, batch_size):
        """标准 SAC 策略更新"""
        # 从 ReplayBuffer 采样
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # --- Critic (LadmCritic) 更新 ---
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            
            # 计算目标 Q 值
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (not_done * self.gamma * (target_q - self.alpha * next_log_pi))

        # 获取当前 Q 估计
        current_q = self.critic(state, action)

        # 计算 MSE 损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor 更新 ---
        pi, log_pi = self.actor.sample(state)
        q_pi = self.critic(state, pi)

        # SAC 目标函数：最大化 (奖励 + 熵)
        actor_loss = ((self.alpha * log_pi) - q_pi).mean()
        
        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- 软更新目标网络 ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """
        保存模型权重。自动创建不存在的目录。
        """
        directory = os.path.dirname(filename)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        print(f">>> 模型保存中: {filename}_*.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        print(">>> 保存完成。")

    def load(self, filename):
        """
        加载模型。支持跨设备加载（如 CUDA -> CPU）。
        """
        print(f">>> 正在从 {filename} 加载权重...")
        # map_location 确保了即使权重是在 GPU 上训练的，也能在当前 device 上正确加载
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=self.device))
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        
        # 同时同步目标网络以保证评估一致性
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(">>> 加载成功。")