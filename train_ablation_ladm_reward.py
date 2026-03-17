import yaml
import numpy as np
import torch
import gymnasium as gym
import highway_env
import collections
from torch.utils.tensorboard import SummaryWriter
import os
import re
import time

# === 关键引入 ===
# 1. 脑子：普通的 MLP Agent (代表传统架构)
from agents.sac_mlp_agent import SAC_MLP_Agent 
from utils.replay_buffer import ReplayBuffer
# 2. 教材：Ladm 物理奖励 (代表物理引导)
from utils.original_ladm_reward import LadmReward 

def get_next_experiment_name(base_dir, base_name="ablation_ladm_reward"):
    """
    自动生成下一个实验的文件夹名称，保持目录结构整洁。
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return f"{base_name}_1"
    
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(base_name)]
    
    if not existing_dirs:
        return f"{base_name}_1"
        
    max_num = 0
    for dir_name in existing_dirs:
        match = re.search(r'_(\d+)$', dir_name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    return f"{base_name}_{max_num + 1}"

def train_ablation():
    # === 1. CUDA 装置设定 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"🔥 [Ablation] 检测到 GPU，将使用: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ [Ablation] 未检测到 GPU，将使用 CPU 进行训练。")

    # 读取通用的配置文件
    with open('configs/sac_ladm_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['device'] = device

    # === 自动化路径管理 (与之前代码保持完全一致) ===
    experiment_name = get_next_experiment_name("./logs", "ablation_ladm_reward")
    print(f"--- 开始 Ablation (MLP + LadmReward) 实验: {experiment_name} ---")

    log_path = f"./logs/{experiment_name}"
    checkpoint_path_base = f"./checkpoints/{experiment_name}"
    trained_model_path_base = f"./trained_models/{experiment_name}"

    # 确保目录存在
    os.makedirs(checkpoint_path_base, exist_ok=True)
    os.makedirs(trained_model_path_base, exist_ok=True)

    writer = SummaryWriter(log_dir=log_path)

    # 环境设定
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "action": {"type": "ContinuousAction"},
        "policy_frequency": 15,
        "crashed_reward": -100, # 这里的数值会被我们的 LadmReward 覆盖
    }
    dt = 1 / env_config["policy_frequency"]
    
    env = gym.make("highway-v0", config=env_config, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    
    # === 2. 设定随机种子 ===
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # === 3. 初始化 Agent: 使用 SAC_MLP_Agent (控制变量: Brain) ===
    agent = SAC_MLP_Agent(state_dim, action_dim, max_action, config)
    
    # 确保模型在 GPU 上
    if hasattr(agent, 'actor'): agent.actor.to(device)
    if hasattr(agent, 'critic'): agent.critic.to(device)
    if hasattr(agent, 'critic_target'): agent.critic_target.to(device)
    if hasattr(agent, 'device'): agent.device = device

    replay_buffer = ReplayBuffer(state_dim, action_dim, config['buffer_size'])
    
    # === 4. 初始化 Ladm 奖励计算器 (控制变量: Reward) ===
    ladm_reward_calculator = LadmReward(dt=dt)

    # === 5. 加载 BC 预训练权重 (控制变量: Initialization) ===
    try:
        bc_model_path = './trained_models/bc_actor_best.pth'
        # 关键：map_location=device 确保直接加载进显存
        agent.actor.load_state_dict(torch.load(bc_model_path, map_location=device))
        print(f"✅ [Ablation] 成功加载 BC 预训练权重: '{bc_model_path}'")
    except FileNotFoundError:
        print(f"⚠️ [Ablation] 警告：找不到 BC 权重 '{bc_model_path}'，将从零开始训练。")

    state, info = env.reset(seed=seed)
    
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_metrics = collections.defaultdict(list)
    last_acceleration = 0.0

    checkpoint_save_freq = 50000 
    best_episode_reward = -np.inf 

    print(f"🚀 [Ablation] 训练开始：MLP 架构 + Ladm 物理奖励...")
    start_time = time.time()
    
    for t in range(int(config['total_timesteps'])):
        episode_timesteps += 1

        if t < config['learning_starts']:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)

# ==========================================================
        # === Ablation 核心逻辑: (已同步为 LADM 冠军版参数) ===
        # ==========================================================
        
        # 1. 获取物理观测数据
        unflattened_obs = env.unwrapped.observation_type.observe()
        
        # 2. 计算 Ladm 瞬时风险
        instantaneous_risk = ladm_reward_calculator.compute_instantaneous_risk(unflattened_obs, action, info)
        
        # 3. 奖励公式 (与 LADM 主实验完全一致，确保公平对比)
        
        # (A) 生存奖励：保持 2.0，确保存活是正向收益
        survival_reward = 2.0 
        
        # (B) 速度奖励：这是之前漏掉的关键项！没有它，模型不会想跑快
        target_speed = 30.0
        speed_reward = 1.0 - abs(info.get('speed', 0) - target_speed) / target_speed
        speed_reward = max(0.0, speed_reward)

        # (C) 风险权重：必须是 0.1！
        # 之前的 5.0 太大了，会导致模型因为扣分太多而选择自杀。
        risk_weight = 0.1 
        weighted_risk = risk_weight * instantaneous_risk

        # (D) 碰撞惩罚：必须是 50.0！
        # 之前的 200.0 会导致模型初期不敢探索。
        collision_penalty = 50.0 if info.get('crashed', False) else 0.0
        
        # 最终奖励公式
        # 预期行为：努力拿满 (生存+速度) 的 3.0 分，同时根据 Risk 微调位置
        ladm_reward = survival_reward + speed_reward - weighted_risk - collision_penalty
        
        # ==========================================================
        
        # 存入 ReplayBuffer 的是 ladm_reward
        replay_buffer.add(state, action, next_state, ladm_reward, done)
        state = next_state
        episode_reward += ladm_reward

        # 指标记录
        episode_metrics["speed"].append(info.get('speed', 0))
        current_acceleration = action[0]
        jerk = (current_acceleration - last_acceleration) / dt
        episode_metrics["acceleration"].append(abs(current_acceleration))
        episode_metrics["jerk"].append(abs(jerk))
        last_acceleration = current_acceleration

        if t >= config['learning_starts']:
            agent.update(replay_buffer, config['batch_size'])

        # === 定期保存 ===
        if (t + 1) % checkpoint_save_freq == 0:
            path = f"{checkpoint_path_base}/step_{t+1}"
            agent.save(path)
            print(f"\n--- [Ablation] 已储存定期检查点: {path}_actor.pth ---")

        if terminated or truncated:
            fps = int(t / (time.time() - start_time))
            print(f"Ablation | 步数: {t+1} | 回合: {episode_num+1} | 奖励: {episode_reward:.2f} | FPS: {fps}")
            
            # --- TensorBoard 记录 (使用 Ablation 标签，方便对比) ---
            writer.add_scalar("Ablation/LadmReward/Episode Reward", episode_reward, episode_num)
            
            crashed = info.get('crashed', False)
            writer.add_scalar("Metrics/Ablation/Collision Rate", 1.0 if crashed else 0.0, episode_num)
            
            success = truncated and not crashed
            writer.add_scalar("Metrics/Ablation/Success Rate", 1.0 if success else 0.0, episode_num)
            
            if episode_metrics["speed"]: 
                writer.add_scalar("Metrics/Ablation/Avg Speed", np.mean(episode_metrics["speed"]), episode_num)
            if episode_metrics["jerk"]: 
                writer.add_scalar("Metrics/Ablation/Avg Jerk", np.mean(episode_metrics["jerk"]), episode_num)

            # === 保存最佳模型 ===
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_path = f"{trained_model_path_base}/best_model"
                agent.save(best_path)
                print(f"*** [Ablation] 新的最佳模型！奖励: {best_episode_reward:.2f} ***")

            # 重置环境
            state, info = env.reset()
            ladm_reward_calculator.reset() # [关键] 重置 Ladm 状态
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_metrics.clear()
            last_acceleration = 0.0

    writer.close()
    final_path = f"{trained_model_path_base}/final_model"
    agent.save(final_path)
    env.close()
    print("Ablation (MLP + LadmReward) 实验训练完成。")

if __name__ == "__main__":
    train_ablation()