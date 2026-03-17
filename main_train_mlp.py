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

def get_next_experiment_name(base_dir, base_name="mlp_baseline_experiment"):
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

def train_mlp_baseline():
    # 1. === CUDA 裝置設定 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True # 對於固定輸入維度的 MLP 非常有效
        print(f"🔥 檢測到 GPU，將使用: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 未檢測到 GPU，將使用 CPU 進行訓練。")

    # 修正: 確保能正確讀取設定檔
    with open('configs/sac_ladm_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 將 device 注入 config，讓 Agent 內部也能讀取到
    config['device'] = device

    # === 自動化路徑管理 ===
    experiment_name = get_next_experiment_name("./logs", "mlp_baseline_experiment")
    print(f"--- 開始新基準模型實驗: {experiment_name} ---")

    log_path = f"./logs/{experiment_name}"
    checkpoint_path_base = f"./checkpoints/{experiment_name}"
    trained_model_path_base = f"./trained_models/{experiment_name}"
    
    # 確保資料夾存在
    os.makedirs(checkpoint_path_base, exist_ok=True)
    os.makedirs(trained_model_path_base, exist_ok=True)

    # 初始化 SummaryWriter
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
        "show_lanes": True,
        "crashed_reward": -5.0, 
    }
    dt = 1 / env_config["policy_frequency"]
    
    env = gym.make("highway-v0", config=env_config, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    
    # 2. === 設定隨機種子 (包含 CUDA) ===
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 3. === 初始化 Agent 並移動到 GPU ===
    agent = SAC_MLP_Agent(state_dim, action_dim, max_action, config)
    
    # 強制將 Agent 內部的神經網路移動到 GPU
    # 這是為了防止 Agent 類別內部沒有自動處理 .to(device) 的情況
    if hasattr(agent, 'actor'): agent.actor.to(device)
    if hasattr(agent, 'critic'): agent.critic.to(device)
    if hasattr(agent, 'critic_target'): agent.critic_target.to(device)
    # 處理 SAC 常見的雙 Critic 架構
    if hasattr(agent, 'critic_1'): agent.critic_1.to(device)
    if hasattr(agent, 'critic_2'): agent.critic_2.to(device)
    if hasattr(agent, 'target_critic_1'): agent.target_critic_1.to(device)
    if hasattr(agent, 'target_critic_2'): agent.target_critic_2.to(device)
    
    # 更新 Agent 內部的 device 屬性 (如果有的話)，這影響 select_action 中的張量轉換
    if hasattr(agent, 'device'):
        agent.device = device

    replay_buffer = ReplayBuffer(state_dim, action_dim, config['buffer_size'])

    state, info = env.reset(seed=config['seed'])
    
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_metrics = collections.defaultdict(list)
    last_acceleration = 0.0

    checkpoint_save_freq = 50000 
    best_episode_reward = -np.inf 

    print(f"開始訓練 SAC-MLP 基準模型 (於 {device} 運行)...")
    
    for t in range(int(config['total_timesteps'])):
        episode_timesteps += 1

        if t < config['learning_starts']:
            action = env.action_space.sample()
        else:
            # Agent 的 select_action 通常接收 numpy，內部轉 tensor(gpu) 再轉回 numpy
            action = agent.select_action(np.array(state))

        next_state, reward, terminated, truncated, info = env.step(action)

        done = float(terminated or truncated)

    # ================== 修改開始 (与 LADM 保持完全一致) ==================
        
        # 1. 生存奖励 (Base Salary)
        # 保持与 LADM 一样的 2.0，确保 Agent 知道"活着"是第一要务
        survival_reward = 2.0
        
        # 2. 速度奖励 (Performance Bonus)
        # 保持与 LADM 一样的逻辑：目标 30km/h，越接近分越高，最高 +1.0
        target_speed = 30.0
        speed_reward = 1.0 - abs(info.get('speed', 0) - target_speed) / target_speed
        speed_reward = max(0.0, speed_reward)

        # 3. 风险项 (Risk Penalty)
        # 这里是 MLP 与 LADM 的关键区别！
        # LADM 有 risk_weight * risk，MLP 没有 Risk 感知能力。
        # 为了公平，MLP 在这里只能拿到 0 (因为它看不到 risk)，这正是它的劣势所在。
        weighted_risk = 0.0 

        # 4. 碰撞惩罚 (Crash Fine)
        # 必须与 LADM 一样设为 50.0，这样两个模型对"死亡"的恐惧程度才是一样的。
        collision_penalty = 50.0 if info.get('crashed', False) else 0.0
        
        # 5. 车道奖励 (Lane Keeping)
        # 保持微小的引导，避免它一直骑线开。
        road = env.unwrapped.road
        vehicle = env.unwrapped.vehicle
        lanes = road.network.lanes_list()
        current_lane_index = vehicle.lane_index[2]
        if len(lanes) > 1:
            right_lane_reward = 0.1 * (current_lane_index / (len(lanes) - 1))
        else:
            right_lane_reward = 0.0

        # 6. 最终公式
        # MLP: Survival(2.0) + Speed(0~1.0) + Lane(0~0.1) - Crash(50)
        # 相比 LADM 少了 "- weighted_risk" 这一项。
        designed_reward = (
            survival_reward + 
            speed_reward + 
            right_lane_reward - 
            weighted_risk - 
            collision_penalty
        )
        # ================== 修改結束 ==================
        replay_buffer.add(state, action, next_state, designed_reward, done)
        state = next_state
        episode_reward += designed_reward

        # === 完整指標記錄 ===
        episode_metrics["speed"].append(info.get('speed', 0))
        current_acceleration = action[0]
        jerk = (current_acceleration - last_acceleration) / dt
        episode_metrics["acceleration"].append(abs(current_acceleration))
        episode_metrics["jerk"].append(abs(jerk))
        last_acceleration = current_acceleration

        if t >= config['learning_starts']:
            # update 函式內部應包含將 batch 數據轉為 tensor 並 .to(device) 的邏輯
            agent.update(replay_buffer, config['batch_size'])

        # === 定期儲存檢查點 ===
        if (t + 1) % checkpoint_save_freq == 0:
            checkpoint_path = f"{checkpoint_path_base}/step_{t+1}"
            agent.save(checkpoint_path)
            print(f"\n--- [基準模型] 已儲存檢查點至: {checkpoint_path}_actor.pth ---")

        if terminated or truncated:
            print(f"基準模型 | 總步數: {t+1} | 回合: {episode_num+1} | 獎勵: {episode_reward:.3f}")
            
            # --- 寫入 TensorBoard ---
            writer.add_scalar("Baseline/MLP/Episode Reward", episode_reward, episode_num)
            writer.add_scalar("Baseline/MLP/Episode Steps", episode_timesteps, episode_num)
            crashed = info.get('crashed', False)
            writer.add_scalar("Metrics/MLP/Collision Rate", 1.0 if crashed else 0.0, episode_num)
            success = truncated and not crashed
            writer.add_scalar("Metrics/MLP/Success Rate", 1.0 if success else 0.0, episode_num)
            if episode_metrics["speed"]: writer.add_scalar("Metrics/MLP/Average Speed (m/s)", np.mean(episode_metrics["speed"]), episode_num)
            if episode_metrics["acceleration"]: writer.add_scalar("Metrics/MLP/Average Acceleration (m/s^2)", np.mean(episode_metrics["acceleration"]), episode_num)
            if episode_metrics["jerk"]: writer.add_scalar("Metrics/MLP/Average Jerk (m/s^3)", np.mean(episode_metrics["jerk"]), episode_num)

            # === 儲存最佳模型 ===
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_model_path = f"{trained_model_path_base}/best_model"
                agent.save(best_model_path)
                print(f"*** [基準模型] 新的最佳模型！獎勵: {best_episode_reward:.3f}。已儲存至: {best_model_path}_actor.pth ***")

            # 重置回合變數
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
    print(f"基準模型訓練完成。最終模型已儲存至 {final_model_path}。")

if __name__ == "__main__":
    train_mlp_baseline()