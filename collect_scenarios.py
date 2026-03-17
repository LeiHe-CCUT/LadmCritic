import os
import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import torch
import yaml
from agents.sac_ladm_agent_new import SAC_Ladm_Agent  # 导入你刚刚命名的类
def collect_data():
    # ================= 1. 路径与配置 =================
    SAVE_DIR = "论文的插图/captured_data"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    config_path = 'configs/sac_ladm_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'lr_actor': 3e-4, 'lr_critic': 3e-4, 'gamma': 0.99, 'tau': 0.005, 'alpha': 0.2}
    
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # ================= 2. 初始化 Agent =================
    state_dim = 35 
    action_dim = 2
    max_action = 1.0
    agent = SAC_Ladm_Agent(state_dim, action_dim, max_action, config)

    # 💡 --- 核心改进：自动寻找可用的模型 ---
    # 请根据你 trained_models 文件夹下的实际名字修改这里的 experiment_folder
    experiment_folder = "ladm_experiment_1" 
    base_path = f"trained_models/{experiment_folder}"
    
    # 依次尝试加载：best_model -> final_model -> 提示失败
    success_load = False
    for model_name in ["best_model", "final_model"]:
        model_path = os.path.join(base_path, model_name)
        if os.path.exists(model_path + "_actor.pth"):
            try:
                agent.load(model_path)
                print(f"✅ 成功加载权重文件: {model_path}")
                success_load = True
                break
            except Exception as e:
                print(f"❌ 加载 {model_name} 失败: {e}")
    
    if not success_load:
        print(f"\n‼️ 错误：在 {base_path} 中找不到任何模型文件！")
        print(f"请检查该文件夹，确认是否存在类似 'best_model_actor.pth' 的文件。")
        print(f"目前的搜索目录是: {os.path.abspath(base_path)}")
        # 如果你想强制退出，取消下面这一行的注释
        # return 

    # ================= 3. 场景配置 =================
    scenarios = [
        {"name": "Scenario_A_CutIn",    "seed": 15, "vehicles": 20, "desc": "紧急切入避障"},
        {"name": "Scenario_B_Following", "seed": 42, "vehicles": 10, "desc": "稳定跟车"},
        {"name": "Scenario_C_Crowded",   "seed": 99, "vehicles": 30, "desc": "多车密集交互"},
        {"name": "Scenario_D_FreeFlow",  "seed": 7,  "vehicles": 5,  "desc": "自由流巡航"}
    ]

    # ================= 4. 开始采集 =================
    for i, cfg in enumerate(scenarios):
        print(f"\n🚀 正在采集场景 {i+1}/4: {cfg['desc']}")
        
        env = gym.make("highway-v0")
        env.unwrapped.config.update({
            "vehicles_count": cfg['vehicles'],
            "duration": 50,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": True,
                "flatten": False
            },
            "action": {"type": "ContinuousAction"},
            "simulation_frequency": 15,
            "policy_frequency": 5
        })
        
        obs, _ = env.reset(seed=cfg['seed'])
        log_data = []

        for t in range(50):
            state_input = obs.flatten()
            
            # 使用 evaluate=True 保证输出不含噪声，曲线更平滑
            action, logic_vars = agent.select_action(state_input, evaluate=True, return_logic=True)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 物理量提取与还原
            ego = obs[0]
            # 寻找视场内最近的威胁车辆
            lead = obs[1] if (len(obs) > 1 and obs[1][0] > 0) else ego

            log_data.append({
                'time': t * 0.2,
                'ego_x': ego[1] * 100,  
                'ego_y': ego[2] * 4,    
                'ego_v': ego[3] * 20,   
                'lead_x': lead[1] * 100,
                'lead_y': lead[2] * 4,
                'ws': logic_vars.get('ws', 0.2), 
                'we': logic_vars.get('we', 0.7),
                'wc': logic_vars.get('wc', 0.1),
                'u_risk': logic_vars.get('u_risk', 10.0),
                'accel': action[0],
                'steer': action[1]
            })

            obs = next_obs
            if terminated or truncated: break
        
        df = pd.DataFrame(log_data)
        file_path = f"{SAVE_DIR}/{cfg['name']}.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ 数据已保存至: {file_path}")
        env.close()

    print("\n🎉 采集完成！请运行 plot_fig6_matrix.py 生成 4x3 矩阵图。")

if __name__ == "__main__":
    collect_data()