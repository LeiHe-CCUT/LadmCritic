import os
import torch
import torch.nn.functional as F
from models.actor import Actor
from models.mlp_critic import MLP_Critic # 匯入我們之前建立的 MLP Critic

class SAC_MLP_Agent:
    """
    使用標準 MLP Critic 的 SAC Agent。
    演算法的更新邏輯遵循標準的 Twin-SAC。
    """
    def __init__(self, state_dim, action_dim, max_action, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

        # 使用 MLP_Critic
        self.critic = MLP_Critic(state_dim, action_dim).to(self.device)
        self.critic_target = MLP_Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # --- Critic (MLP_Critic) 更新 ---
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            
            # 計算目標 Q 值 (取兩個目標 Q 網路中的較小值)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (not_done * self.gamma * (target_q - self.alpha * next_log_pi))

        # 獲取當前 Q 值的估計
        current_q1, current_q2 = self.critic(state, action)

        # 計算 Critic 損失 (兩個 Q 網路的損失之和)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor 更新 ---
        # 為了提高效率，Actor 的更新不需要計算梯度
        for p in self.critic.parameters():
            p.requires_grad = False

        pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True
        
        # --- 目標網路的軟更新 ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic_target.load_state_dict(self.critic.state_dict())