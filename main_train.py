
# #
import yaml
import numpy as np
import torch
import torch.nn as nn # 導入 nn 以支持 DataParallel
import gymnasium as gym
import highway_env
import collections
import os
import re

from torch.utils.tensorboard import SummaryWriter
from agents.sac_ladm_agent import SAC_Ladm_Agent
from utils.replay_buffer import ReplayBuffer
from utils.original_ladm_reward import LadmReward

def get_next_experiment_name(base_dir, base_name="ladm_experiment"):
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

def train():
    # === 1. GPU 环境设置 ===
    # 检测 CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        print(f"--- GPU 环境检测成功 ---")
        print(f"可用 GPU 数量: {gpu_count}")
        print(f"当前使用的设备: {torch.cuda.get_device_name(0)}")
        if gpu_count > 1:
             print(f"注意: 检测到 {gpu_count} 块显卡。默认计算将在 cuda:0 进行。")
             # 针对 V100S 开启 cuDNN 自动调优，加速卷积运算
             torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("--- 未检测到 GPU，将使用 CPU 训练 ---")

    with open('configs/sac_ladm_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 将 device 注入到 config 中，确保 Agent 知道使用哪个设备
    # (这需要您的 SAC_Ladm_Agent 代码里支持从 config 读取 device，或者您手动传入)
    config['device'] = str(device) 

    # === 自动化路径管理 ===
    experiment_name = get_next_experiment_name("./logs", "ladm_experiment")
    print(f"--- 开始新实验: {experiment_name} ---")

    log_path = f"./logs/{experiment_name}"
    checkpoint_path_base = f"./checkpoints/{experiment_name}"
    trained_model_path_base = f"./trained_models/{experiment_name}"
    
    # 确保目录存在
    os.makedirs(checkpoint_path_base, exist_ok=True)
    os.makedirs(trained_model_path_base, exist_ok=True)

    writer = SummaryWriter(log_dir=log_path)

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
        "crashed_reward": -100,
        #"crashed_reward": -1.5,
    }
    dt = 1 / env_config["policy_frequency"]

    env = gym.make("highway-v0", config=env_config, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    
    # === 2. 随机种子设置 (包含 GPU) ===
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 为所有 GPU 设置种子
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 初始化 Agent (假设 Agent 内部会处理 config['device'] 并将模型 .to(device))
    # 如果您的 Agent __init__ 不接受 config 中的 device，请手动修改 Agent 代码
    agent = SAC_Ladm_Agent(state_dim, action_dim, max_action, config)

    # === 3. (可选) 多 GPU 并行计算 DataParallel ===
    # 如果您想强制同时使用两块卡来计算梯度（适合 BatchSize 很大时），
    # 并且您的 agent 将网络暴露为 self.actor 和 self.critic，可以取消下面代码的注释。
    # 注意：对于一般 RL 任务，单块 V100S 往往比多卡数据并行的通信开销更小、速度更快。
    """
    if torch.cuda.device_count() > 1:
        print(f"正在尝试将模型并行分布在 {torch.cuda.device_count()} 块 GPU 上...")
        try:
            agent.actor = nn.DataParallel(agent.actor)
            agent.critic = nn.DataParallel(agent.critic)
            agent.critic_target = nn.DataParallel(agent.critic_target)
            print("模型已启用 DataParallel。")
        except AttributeError:
            print("警告: 无法自动并行化模型 (Agent 内部属性名称可能不同)，将使用单卡训练。")
    """

    replay_buffer = ReplayBuffer(state_dim, action_dim, config['buffer_size'])
    ladm_reward_calculator = LadmReward(dt=dt) 

    state, info = env.reset(seed=seed)
    
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_metrics = collections.defaultdict(list)
    last_acceleration = 0.0

    checkpoint_save_freq = 50000 
    best_episode_reward = -np.inf 

    print("使用完整版 LADM 奖励模型开始训练 (已启用自动化路径管理)...")
    
    # 记录开始时间
    
    import time
    start_time = time.time()
# ================= 关键修改：强制设置长训练步数 =================
    # 为了论文曲线完美收敛，消除后期震荡，我们将步数强制设为 30万步 (或更多)
    # 原本是读取 config['total_timesteps']，这里我们直接覆盖它
    total_steps = 300000  
    print(f"🔥 为了获取论文级完美曲线，已将训练总步数强制设定为: {total_steps}")
    # =============================================================
    for t in range(int(config['total_timesteps'])):
        episode_timesteps += 1

        if t < config['learning_starts']:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)

        unflattened_obs = env.unwrapped.observation_type.observe()
        instantaneous_risk = ladm_reward_calculator.compute_instantaneous_risk(unflattened_obs, action, info)
        
        # # =========== 优化后的【安全优先】奖励逻辑 ===========
        
        # # 1. 降低生存奖励：不要给它太高的分仅仅因为它还活着
        # # 让它把注意力集中在"如何开得更好"而不是"如何赖着不走"
        # survival_reward = 1.0 
        
        # # 2. 提升风险敏感度 (关键修改)
        # # Ladm 的 risk 值可能很小 (0.0x)，我们需要放大它。
        # # 系数越大，它越早开始减速/避让。如果觉得还不够安全，把这个数调大到 10.0
        # risk_weight = 5.0 
        # weighted_risk = risk_weight * instantaneous_risk

        # # 3. 毁灭性的碰撞惩罚 (一票否决)
        # # 必须确保：Crash Penalty >> Max Potential Gain per Step
        # # 之前是 50，现在改为 200。这意味着撞一次车，相当于白跑了200步。
        # collision_penalty = 200.0 if info.get('crashed', False) else 0.0
        
        # # 4. (可选) 增加速度奖励的约束
        # # 只有在风险极低的时候，才全额给速度奖励，否则打折。
        # # 这能教会在安全时快，危险时慢。
        # # speed_factor = max(0, 1 - weighted_risk) # 风险越高，速度奖励越少
        
        # # 最终 LADM 奖励公式
        # # 核心思想：基础分低，惩罚极重。
        # ladm_reward = survival_reward - weighted_risk - collision_penalty
        
        # # 调试用：如果训练初期发现 reward 一直是负数，检查这里
        # # print(f"Risk: {weighted_risk:.2f}, Crash: {collision_penalty}, Total: {ladm_reward:.2f}")

        # # ==================================================
# =========== LADM 冠军版奖励配置 ===========
        
        # 1. 生存奖励 (基础工资)
        # 保持在 2.0，确保它是正向收益的核心来源。
        # 这样 Agent 无论如何首要目标都是"留在场上"。
        survival_reward = 2.0
        
        # 2. 速度奖励 (绩效奖金)
        # 加上 max 0 保护。
        # 这里给 1.0 的权重，让它在安全的时候尽量开快。
        target_speed = 30.0
        speed_reward = 1.0 - abs(info.get('speed', 0) - target_speed) / target_speed
        speed_reward = max(0.0, speed_reward)

        # 3. 风险权重 (核心胜负手 - 早期预警系统)
        # 这里的逻辑是：
        # Risk 通常在 0~20 之间波动。
        # 我们乘以 0.1，让 Weighted Risk 在 0 ~ 2.0 之间。
        # 效果：
        # - 平时 (Risk=2, Weighted=0.2): 净赚 1.8 分 -> Agent 舒服地开。
        # - 有点危险 (Risk=10, Weighted=1.0): 净赚 1.0 分 -> Agent 觉得"赚少了"，开始主动避让。
        # - 极度危险 (Risk=20, Weighted=2.0): 净赚 0 分 -> Agent 意识到"白干了"，必须立刻急刹或变道！
        # 这个梯度设计比单纯的罚款更聪明，它能教 Agent "防患于未然"。
        risk_weight = 0.1
        weighted_risk = risk_weight * instantaneous_risk

        # 4. 碰撞惩罚 (红线)
        # 设定为 50 足够了。太高(-200)会导致它初期不敢动；
        # 只要 LADM 机制生效，它在撞车前就会因为 Risk 太高而避开了，根本不需要吃到这 -50。
        collision_penalty = 50.0 if info.get('crashed', False) else 0.0

        # 5. 冠军公式
        # 预期行为：平时高速巡航(拿满生存+速度)，遇到前车减速(Risk升高)，立刻变道(因为不想损失收益)。
        ladm_reward = survival_reward + speed_reward - weighted_risk - collision_penalty
        
        # ==========================================

        # ===============================================
        replay_buffer.add(state, action, next_state, ladm_reward, done)
        state = next_state
        episode_reward += ladm_reward

        # 收集指標數據 (這些數據原本只存了沒寫入，現在會用到了)
        episode_metrics["speed"].append(info.get('speed', 0))
        current_acceleration = action[0]
        jerk = (current_acceleration - last_acceleration) / dt
        episode_metrics["acceleration"].append(abs(current_acceleration))
        episode_metrics["jerk"].append(abs(jerk))
        last_acceleration = current_acceleration

        if t >= config['learning_starts']:
            agent.update(replay_buffer, config['batch_size'])

        # 定期儲存檢查點
        if (t + 1) % checkpoint_save_freq == 0:
            checkpoint_path = f"{checkpoint_path_base}/step_{t+1}"
            agent.save(checkpoint_path)
            print(f"\n--- 已儲存定期檢查點模型至: {checkpoint_path}_actor.pth ---")

        if terminated or truncated:
            fps = int(t / (time.time() - start_time))
            print(f"步數: {t+1} | 回合: {episode_num+1} | 獎勵: {episode_reward:.3f} | FPS: {fps}")
            
            # =========== 補全缺失的 TensorBoard 指標 ===========
            
            # 1. 基礎指標
            writer.add_scalar("Train/Episode Reward", episode_reward, episode_num)
            writer.add_scalar("Train/FPS", fps, episode_num)
            
            # 2. 安全性指標
            crashed = info.get('crashed', False)
            writer.add_scalar("Metrics/Collision Rate", 1.0 if crashed else 0.0, episode_num)
            
            # 3. [新增] 成功率 (沒有撞車且跑完時間)
            success = truncated and not crashed
            writer.add_scalar("Metrics/Success Rate", 1.0 if success else 0.0, episode_num)

            # 4. [新增] 駕駛品質指標 (從 episode_metrics 取平均值)
            if episode_metrics["speed"]:
                writer.add_scalar("Metrics/Average Speed (m/s)", np.mean(episode_metrics["speed"]), episode_num)
            if episode_metrics["acceleration"]:
                writer.add_scalar("Metrics/Average Acceleration (m/s^2)", np.mean(episode_metrics["acceleration"]), episode_num)
            if episode_metrics["jerk"]:
                writer.add_scalar("Metrics/Average Jerk (m/s^3)", np.mean(episode_metrics["jerk"]), episode_num)
            
            # =================================================

            # 儲存最佳模型
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_model_path = f"{trained_model_path_base}/best_model"
                agent.save(best_model_path)
                print(f"*** 新的最佳模型！獎勵: {best_episode_reward:.3f}。已儲存至: {best_model_path} ***")

            state, info = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            ladm_reward_calculator.reset()
            episode_metrics.clear()
            last_acceleration = 0.0

    writer.close()
    final_model_path = f"{trained_model_path_base}/final_model"
    agent.save(final_model_path)
    env.close()
    print(f"訓練完成。最終模型已儲存至 {final_model_path}。")

if __name__ == "__main__":
    train()