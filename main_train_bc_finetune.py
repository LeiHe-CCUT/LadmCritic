
import yaml
import numpy as np
import torch
import gymnasium as gym
import highway_env
import collections
from torch.utils.tensorboard import SummaryWriter
import os
import re

# 匯入 MLP 版本的 Agent
from agents.sac_mlp_agent import SAC_MLP_Agent
from utils.replay_buffer import ReplayBuffer

def get_next_experiment_name(base_dir, base_name="bc_finetune_experiment"):
    """
    自動生成下一個實驗的檔案夾名稱。
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

def train_bc_finetune():
    # 1. === CUDA 裝置設定 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True # 加速固定輸入大小的網路運算
        print(f"🔥 檢測到 GPU，將使用: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 未檢測到 GPU，將使用 CPU 進行訓練。")

    with open('configs/sac_ladm_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 將 device 加入 config，以便 Agent 內部可以使用
    config['device'] = device

    # === 自動化路徑管理 ===
    experiment_name = get_next_experiment_name("./logs", "bc_finetune_experiment")
    print(f"--- 開始新 BC 微調實驗: {experiment_name} ---")

    log_path = f"./logs/{experiment_name}"
    checkpoint_path_base = f"./checkpoints/{experiment_name}"
    trained_model_path_base = f"./trained_models/{experiment_name}"

    # 確保目錄存在
    os.makedirs(checkpoint_path_base, exist_ok=True)
    os.makedirs(trained_model_path_base, exist_ok=True)

    writer = SummaryWriter(log_dir=log_path)

    # 環境設定
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "action": {"type": "ContinuousAction"},
        "policy_frequency": 15,
        "crashed_reward": -5.0,
    }
    dt = 1 / env_config["policy_frequency"]
    
    env = gym.make("highway-v0", config=env_config, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    
    # 設定隨機種子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 2. === 實例化 MLP Agent 並移至 GPU ===
    # 這裡假設您的 SAC_MLP_Agent 構造函數接受 device 參數，或者我們在下方手動移動
    try:
        agent = SAC_MLP_Agent(state_dim, action_dim, max_action, config) # 如果 config 內含 device，通常 Agent 會自動處理
    except TypeError:
        # 如果 Agent 不接受 config 內的 device，我們使用標準初始化
        agent = SAC_MLP_Agent(state_dim, action_dim, max_action, config)
    
    # 確保 Agent 內部的神經網路都在 GPU 上
    # 假設 Agent 有 actor, critic, critic_target 等成員
    if hasattr(agent, 'actor'): agent.actor.to(device)
    if hasattr(agent, 'critic'): agent.critic.to(device) 
    if hasattr(agent, 'critic_target'): agent.critic_target.to(device)
    # 如果有其他的 target 網路 (例如 SAC 雙 Critic)
    if hasattr(agent, 'critic_1'): agent.critic_1.to(device)
    if hasattr(agent, 'critic_2'): agent.critic_2.to(device)
    if hasattr(agent, 'target_critic_1'): agent.target_critic_1.to(device)
    if hasattr(agent, 'target_critic_2'): agent.target_critic_2.to(device)
    
    # 將 Agent 內部的 device 屬性更新 (如果有的話)，確保 select_action 能正確轉換張量
    if hasattr(agent, 'device'):
        agent.device = device

    replay_buffer = ReplayBuffer(state_dim, action_dim, config['buffer_size'])

    # 3. === 載入預訓練權重 (修正 map_location) ===
    try:
        bc_model_path = './trained_models/bc_actor_best.pth'
        # 關鍵：這裡使用 map_location=device 確保權重直接載入到正確的設備
        agent.actor.load_state_dict(torch.load(bc_model_path, map_location=device))
        print(f"成功從 '{bc_model_path}' 載入行為克隆預訓練的 Actor 權重！")
    except FileNotFoundError:
        print(f"警告：找不到預訓練的 Actor 模型 '{bc_model_path}'。將從零開始訓練。")
    except RuntimeError as e:
        print(f"載入權重時發生錯誤 (可能是維度不匹配): {e}")

    state, info = env.reset(seed=config['seed'])
    
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_metrics = collections.defaultdict(list)
    last_acceleration = 0.0

    checkpoint_save_freq = 50000 
    best_episode_reward = -np.inf 

    print(f"開始訓練 BC + SAC-MLP 基準模型 (微調) 於 {device}...")
    
    for t in range(int(config['total_timesteps'])):
        episode_timesteps += 1

        if t < config['learning_starts']:
            action = env.action_space.sample()
        else:
            # Agent.select_action 通常會處理 numpy -> tensor(gpu) -> numpy
            action = agent.select_action(np.array(state))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)


        # ================== 修改起始位置 ==================
        
        # 1. 速度奖励优化：加上 max(0, ...) 保护，防止超速过多变成负分
        target_speed = 30.0
        speed_reward = 1.0 - abs(info.get('speed', 0) - target_speed) / target_speed
        speed_reward = max(0.0, speed_reward) 

        # 2. 生存奖励：只要活着就给分 (保持不变)
        alive_reward = 0.5 

        # 3. 车道奖励 (保持不变)
        road = env.unwrapped.road
        vehicle = env.unwrapped.vehicle
        lanes = road.network.lanes_list()
        current_lane_index = vehicle.lane_index[2]
        if len(lanes) > 1:
            right_lane_reward = current_lane_index / (len(lanes) - 1)
        else:
            right_lane_reward = 0.0
        
        # 4. 关键修改：大幅提高碰撞惩罚！
        # 原代码是 10.0。鉴于您的总分高达 1300+，扣 10 分对 Agent 来说微不足道。
        # 建议提高到 50.0，让它彻底不敢撞车。
        collision_penalty = 50.0 if info.get('crashed', False) else 0.0

        # 组合奖励
        designed_reward = (
            speed_reward + 
            alive_reward + 
            right_lane_reward - 
            collision_penalty
        )
        # ================== 修改结束位置 ==================

   
        
        replay_buffer.add(state, action, next_state, designed_reward, done)
        state = next_state
        episode_reward += designed_reward

        # 指標記錄
        episode_metrics["speed"].append(info.get('speed', 0))
        current_acceleration = action[0]
        jerk = (current_acceleration - last_acceleration) / dt
        episode_metrics["acceleration"].append(abs(current_acceleration))
        episode_metrics["jerk"].append(abs(jerk))
        last_acceleration = current_acceleration

        if t >= config['learning_starts']:
            # Agent.update 內部應該負責從 ReplayBuffer 取樣並轉為 GPU Tensor
            agent.update(replay_buffer, config['batch_size'])

        if (t + 1) % checkpoint_save_freq == 0:
            checkpoint_path = f"{checkpoint_path_base}/step_{t+1}"
            agent.save(checkpoint_path)
            print(f"\n--- [BC+SAC] 已儲存檢查點至: {checkpoint_path}_actor.pth ---")

        if terminated or truncated:
            print(f"BC+SAC 微調 | 總步數: {t+1} | 回合: {episode_num+1} | 獎勵: {episode_reward:.3f}")
            
            writer.add_scalar("Baseline/BC_Finetune/Episode Reward", episode_reward, episode_num)
            writer.add_scalar("Baseline/BC_Finetune/Episode Steps", episode_timesteps, episode_num)
            crashed = info.get('crashed', False)
            writer.add_scalar("Metrics/BC_Finetune/Collision Rate", 1.0 if crashed else 0.0, episode_num)
            success = truncated and not crashed
            writer.add_scalar("Metrics/BC_Finetune/Success Rate", 1.0 if success else 0.0, episode_num)
            
            if episode_metrics["speed"]: 
                writer.add_scalar("Metrics/BC_Finetune/Average Speed (m/s)", np.mean(episode_metrics["speed"]), episode_num)
            if episode_metrics["acceleration"]: 
                writer.add_scalar("Metrics/BC_Finetune/Average Acceleration (m/s^2)", np.mean(episode_metrics["acceleration"]), episode_num)
            if episode_metrics["jerk"]: 
                writer.add_scalar("Metrics/BC_Finetune/Average Jerk (m/s^3)", np.mean(episode_metrics["jerk"]), episode_num)

            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_model_path = f"{trained_model_path_base}/best_model"
                agent.save(best_model_path)
                print(f"*** [BC+SAC] 新的最佳模型！獎勵: {best_episode_reward:.3f}。已儲存至: {best_model_path}_actor.pth ***")

            state, info = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_metrics.clear()
            last_acceleration = 0.0

    writer.close()
    final_model_path = f"{trained_model_path_base}/final_model"
    agent.save(final_model_path)
    env.close()
    print(f"BC+SAC 微調訓練完成。最終模型已儲存至 {final_model_path}。")

if __name__ == "__main__":
    train_bc_finetune()