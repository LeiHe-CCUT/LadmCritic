# agents/sac_ladm_agent.py
import torch
import torch.nn.functional as F
from models.actor import Actor
from models.ladm_critic import LadmCritic
import os

class SAC_Ladm_Agent:
    def __init__(self, state_dim, action_dim, max_action, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Actor network
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

        # Create LadmCritic network (as described in the paper)
        self.critic = LadmCritic(state_dim, action_dim, config).to(self.device)
        self.critic_target = LadmCritic(state_dim, action_dim, config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        
        # SAC parameters
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size):
        # Sample a batch from the replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # --- Critic (LadmCritic) Update ---
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            
            # Compute the target Q value
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (not_done * self.gamma * (target_q - self.alpha * next_log_pi))

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        pi, log_pi = self.actor.sample(state)
        q_pi = self.critic(state, pi)

        actor_loss = ((self.alpha * log_pi) - q_pi).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- Soft update for target networks ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filename):
        """保存所有模型的权重。"""
        print(f"正在保存模型至: {filename}_*.pth")
        
        # --- (核心修正) ---
        # 1. 从完整文件路径中获取目录路径
        directory = os.path.dirname(filename)

        # 2. 如果目录不存在，则创建它
        # exist_ok=True 确保了即使目录已经存在，代码也不会报错
        os.makedirs(directory, exist_ok=True)
        
        # 3. 现在可以安全地保存文件了
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        # 如果您还有其他模型，也在这里保存
        # torch.save(self.other_model.state_dict(), filename + "_other.pth")

        print("模型保存成功。")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        # Load target network as well for evaluation consistency
        self.critic_target = self.critic