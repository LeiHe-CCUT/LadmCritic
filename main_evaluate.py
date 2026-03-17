import yaml
import numpy as np
import torch
import gymnasium as gym
import highway_env
import pandas as pd
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import re
import warnings

# --- 忽略不必要的警告 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

from models.actor import Actor
from models.ladm_critic import LadmCritic
from agents.rule_based_agents import AggressiveAgent
from utils.original_ladm_reward import LadmReward

def natural_sort_key(s):
    """
    實現自然排序：將字符串中的數字部分解析為整數進行比較
    例如：ladm_experiment_2 會排在 ladm_experiment_10 前面
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def find_experiments(base_dir, prefix=""):
    if not os.path.exists(base_dir):
        return []
    valid_experiments = []
    for d in os.listdir(base_dir):
        if d.startswith(prefix):
            exp_path = os.path.join(base_dir, d)
            # 確保是目錄且包含模型文件
            check_file = 'best_model_actor.pth' if 'ladm' not in prefix or 'critic' not in prefix else 'best_model_critic.pth'
            if os.path.isdir(exp_path): 
                # 寬鬆檢查：只要目錄存在即可，具體文件在加載時再報錯，避免漏掉某些只有部分文件的實驗
                valid_experiments.append(d)
    
    # 使用自然排序
    return sorted(valid_experiments, key=natural_sort_key)

def find_latest_experiment(experiments, prefix):
    relevant_experiments = [e for e in experiments if e.startswith(prefix)]
    if not relevant_experiments:
        return None
    # 使用自然排序找最後一個
    latest_exp = sorted(relevant_experiments, key=natural_sort_key, reverse=True)
    return latest_exp[0] if latest_exp else None

def select_from_menu(options, prompt):
    if not options:
        print("未找到任何可用的實驗記錄。")
        return None
        
    print(f"\n{prompt}")
    # 預設選項通常是最後一個（最新的）
    default_index = len(options) 
    
    for i, name in enumerate(options):
        is_latest = "(最新)" if i == len(options) - 1 else ""
        print(f"  [{i+1}] {name} {is_latest}")
        
    while True:
        try:
            user_input = input(f"請輸入選擇的序號 (預設 {default_index}): ").strip()
            
            # 支援直接按 Enter 選最新的
            if user_input == "":
                choice = default_index
            else:
                choice = int(user_input)
                
            if 1 <= choice <= len(options):
                selected = options[choice - 1]
                print(f"-> 已選擇: {selected}")
                return selected
            else:
                print(f"輸入無效，請輸入 1 到 {len(options)} 之間的數字。")
        except ValueError:
            print("請輸入一個有效的數字。")

def calculate_ttc(obs):
    """計算碰撞時間 (Time To Collision)"""
    # obs[1] 通常是前車。 索引依據 Kinematics 設定：[presence, x, y, vx, vy, ...]
    if obs[1, 0] == 0:  # presence
        return 100.0
    rel_x = obs[1, 1]
    rel_vx = obs[1, 3]
    
    # 只有當前車比我慢（相對速度為負）且在我前面時才計算 TTC
    if rel_vx < -0.1 and rel_x > 0:
        return rel_x / (-rel_vx)
    return 100.0

def evaluate(args):
    print("="*50)
    print("開始 Phase 2: 雙裁判離線評估 (LadmCritic & Static Ladm)")
    print("="*50)
    
    # 1. 選擇裁判模型
    base_model_dir = './trained_models'
    available_ladm_exps = find_experiments(base_model_dir, prefix="ladm_experiment")
    
    evaluator_exp_name = select_from_menu(available_ladm_exps, "請選擇要用作『評估器 (裁判)』的 LadmCritic 訓練實驗:")
    if not evaluator_exp_name:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f"./analysis/evaluation_runs/{timestamp}_judge_{evaluator_exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n本次評估結果將儲存至: {output_dir}")

    # 2. 環境與模型配置
    # 嘗試加載配置，如果失敗則使用預設
    config_path = 'configs/sac_ladm_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"警告: 找不到 {config_path}，使用預設空配置。")
        config = {}

    env_config = {
        "observation": {
            "type": "Kinematics", 
            "vehicles_count": 5, 
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"], 
            "absolute": False
        },
        "action": {"type": "ContinuousAction"}, 
        "policy_frequency": 15,
        "duration": 40 # 增加模擬時長以觀察更完整的行為
    }
    dt = 1 / env_config["policy_frequency"]
    render_mode = "human" if args.render else "rgb_array"
    
    env = gym.make("highway-v0", config=env_config, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 3. 載入「神級裁判」(LadmCritic)
    try:
        frozen_ladm_critic = LadmCritic(state_dim, action_dim, config).to(device)
        critic_path = os.path.join(base_model_dir, evaluator_exp_name, "best_model_critic.pth")
        frozen_ladm_critic.load_state_dict(torch.load(critic_path, map_location=device))
        frozen_ladm_critic.eval()
        print(f"✔ 評估器 LadmCritic 載入成功: {critic_path}")
    except FileNotFoundError:
        print(f"❌ 錯誤: 在 {evaluator_exp_name} 中找不到 best_model_critic.pth。請檢查該實驗是否訓練完成。")
        return

    # 4. 實例化「普通裁判」(靜態 Ladm 模型)
    static_ladm_evaluator = LadmReward(dt=dt)

    # 5. 自動尋找並準備所有待評估的「選手」
    policies_to_evaluate = {}
    all_available_policies = find_experiments(base_model_dir)
    
    # 定義要尋找的策略類型前綴
    policy_types = {
        "Ladm (專家)": "ladm_experiment", 
        "MLP (基準)": "mlp_baseline_experiment", 
        "BC+SAC (微調)": "bc_finetune_experiment"
    }
    
    print("\n自動尋找各類型的最新『選手』模型...")
    for display_name, prefix in policy_types.items():
        latest_policy_exp = find_latest_experiment(all_available_policies, prefix)
        if latest_policy_exp:
            try:
                actor = Actor(state_dim, action_dim, max_action).to(device)
                actor_path = os.path.join(base_model_dir, latest_policy_exp, "best_model_actor.pth")
                actor.load_state_dict(torch.load(actor_path, map_location=device))
                actor.eval()
                policies_to_evaluate[display_name] = actor
                print(f"  Target: [{display_name}] -> 使用實驗: {latest_policy_exp}")
            except FileNotFoundError:
                print(f"  Skipped: [{display_name}] 找到實驗目錄但缺失模型文件 ({latest_policy_exp})")
        else:
            print(f"  Skipped: [{display_name}] 未找到以此為前綴的實驗")
            
    policies_to_evaluate["魯莽駕駛 (Rule-Based)"] = AggressiveAgent(env)
    
    if not policies_to_evaluate:
        print("沒有找到任何可評估的模型！請檢查 trained_models 資料夾。")
        return

    # 6. 開始評估循環
    results = []
    num_episodes = args.episodes
    
    print(f"\n開始評估流程 (每模型 {num_episodes} 回合)...")
    
    for policy_name, policy in policies_to_evaluate.items():
        print(f"正在評估: {policy_name}")
        
        crash_count = 0
        total_steps = 0
        
        for i in tqdm(range(num_episodes), desc=f"Runs ({policy_name})"):
            flattened_obs, info = env.reset()
            static_ladm_evaluator.reset()
            done = truncated = False
            episode_steps = 0
            
            while not done and not truncated:
                # 獲取未壓平的觀測值用於規則計算
                unflattened_obs = env.unwrapped.observation_type.observe()
                
                # 決策動作
                if isinstance(policy, AggressiveAgent):
                    action = policy.act(unflattened_obs)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(flattened_obs).unsqueeze(0).to(device)
                        action_tensor, _ = policy(state_tensor)
                        action = action_tensor.cpu().numpy().flatten()
                
                # === 核心：同時獲取兩個裁判的評分 ===
                # 1. 學習型裁判 (Critic) 打分 (Q-value)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(flattened_obs).unsqueeze(0).to(device)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
                    # LadmCritic 輸出通常是 (Batch, 1)
                    q_ladm_critic_score = frozen_ladm_critic(state_tensor, action_tensor).item()
                
                # 2. 物理型裁判 (Static Ladm) 打分 (Risk -> Reward)
                # 注意：Static Ladm 計算的是瞬時風險 R_total，轉成 Reward 通常是 -R 或者 1/(1+R)
                # 這裡假設我們要記錄的是負風險 (數值越大越安全)
                instantaneous_risk_D = static_ladm_evaluator.compute_instantaneous_risk(unflattened_obs, action, info)
                static_ladm_score = -instantaneous_risk_D
                
                ttc = calculate_ttc(unflattened_obs)
                
                results.append({
                    'policy': policy_name, 
                    'episode': i, 
                    'step': episode_steps,
                    'q_ladm_critic_score': q_ladm_critic_score, 
                    'static_ladm_score': static_ladm_score, 
                    'ttc': ttc
                })
                
                flattened_obs, reward, done, truncated, info = env.step(action)
                episode_steps += 1
                total_steps += 1
                
                if info.get('crashed', False):
                    crash_count += 1
                    
        print(f"  -> 碰撞次數: {crash_count}/{num_episodes}")
    
    # 7. 儲存結果
    results_df = pd.DataFrame(results)
    output_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(output_csv_path, index=False)
    
    print(f"\n評估完成！所有詳細數據已儲存至: {output_csv_path}")
    print(f"你可以使用 analysis/plot_results.py (如果有的話) 來視覺化這些分數。")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: 離線評估駕駛決策模型")
    parser.add_argument("--render", action="store_true", help="是否渲染環境視覺化介面")
    parser.add_argument("--episodes", type=int, default=10, help="每個模型評估的回合數 (預設: 10)")
    args = parser.parse_args()
    evaluate(args)