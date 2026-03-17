import os
import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 配置 =================
ENV_NAME = "highway-v0"
SAVE_DIR = "论文的插图"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_action_and_weights(obs):
    """
    接入模型逻辑。这里演示随机波动效果，
    实际使用请替换为: action, _states = model.predict(obs)
    """
    action = 1 # 保持直行
    # 模拟真实环境下会随路况跳动的权重数据
    ws = 0.2 + np.random.normal(0, 0.02)
    we = 0.7 + np.random.normal(0, 0.02)
    u_risk = 10 + np.random.normal(0, 1.5)
    return action, ws, we, u_risk

# ================= 2. 修复后的真实环境运行与录制 =================
def record_real_scenario(num_steps=60):
    print(f">>> 正在启动 {ENV_NAME} 并录制真实轨迹数据...")
    env = gym.make(ENV_NAME)
    
    # --- 修复核心：使用 .unwrapped 访问 config ---
    env.unwrapped.config.update({
        "duration": num_steps,
        "lanes_count": 4,
        "vehicles_count": 20,
        "observation": {
            "type": "Kinematics", # 确保获取到的是物理坐标
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,    # 使用绝对坐标方便绘图
            "flatten": False
        }
    })
    # 修改配置后需重新 reset
    obs = env.reset()
    
    trajectory_log = []
    
    print(">>> 环境已就绪，开始执行步进记录...")
    for t in range(num_steps):
        # 1. 模型预测
        action, ws, we, u_risk = get_action_and_weights(obs)
        
        # 2. 环境执行
        obs, reward, done, info = env.step(action)
        
        # 3. 提取真实物理量 (obs[0] 为自车, obs[1] 为最近的他车)
        ego = obs[0] 
        lead = obs[1] if len(obs) > 1 else obs[0]
        
        trajectory_log.append({
            'time': t * 0.1,
            'ego_x': ego[1],  # 绝对坐标 x
            'ego_y': ego[2],  # 绝对坐标 y
            'ego_vx': ego[3],
            'chal_x': lead[1],
            'chal_y': lead[2],
            'ws': ws,
            'we': we,
            'u_risk': u_risk
        })
        
        if done: break
    
    env.close()
    df = pd.DataFrame(trajectory_log)
    df.to_csv(os.path.join(SAVE_DIR, "real_scenario_case.csv"), index=False)
    print(f">>> 数据录制完成，共记录 {len(df)} 帧。")
    return df

# ================= 3. 绘图函数 (与之前一致，增强了稳定性) =================
def plot_figure_6_real(df):
    if df.empty: return
    print(">>> 正在渲染 Figure 6 真实轨迹与权重图...")
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])

    # --- 左侧：场景轨迹图 ---
    ax_bev = fig.add_subplot(gs[0])
    # 绘制车道 (highway-v0 车道线通常在 0, 4, 8, 12 等位置)
    for y in [-2, 2, 6, 10, 14]:
        ax_bev.axhline(y, color='#bdc3c7', linestyle='--', linewidth=1, zorder=1)
    
    ax_bev.plot(df['ego_x'], df['ego_y'], color='#d62728', linewidth=3, label='Ego (LadmCritic)', zorder=5)
    ax_bev.scatter(df['chal_x'], df['chal_y'], color='#2c3e50', s=15, alpha=0.3, label='Surrounding Vehicles')

    # 绘制车辆盒子 (t=0, 中间点, 最后点)
    indices = [0, len(df)//2, len(df)-1]
    for i in indices:
        # 自车红色
        ax_bev.add_patch(patches.Rectangle((df['ego_x'][i]-2.2, df['ego_y'][i]-1), 4.4, 2, color='#d62728', alpha=0.9, zorder=6))
        # 他车灰色
        ax_bev.add_patch(patches.Rectangle((df['chal_x'][i]-2.2, df['chal_y'][i]-1), 4.4, 2, color='#95a5a6', alpha=0.7, zorder=4))
        ax_bev.text(df['ego_x'][i], df['ego_y'][i]-3, f"{df['time'][i]:.1f}s", ha='center', fontweight='bold', fontsize=9)

    ax_bev.set_title("Qualitative Case Study: Real Scenario Trace", fontsize=16, fontweight='bold', pad=15)
    ax_bev.set_xlabel("Longitudinal Distance [m]", fontsize=14, fontweight='bold')
    ax_bev.set_ylabel("Lateral Position [m]", fontsize=14, fontweight='bold')
    ax_bev.legend(loc='upper left', frameon=True, shadow=True)

    # --- 右侧：逻辑指标演化图 ---
    ax_logic = fig.add_subplot(gs[1])
    ax_twin = ax_logic.twinx()
    
    ax_logic.plot(df['time'], df['ws'], color='#d62728', linewidth=2.5, label=r'$w_s$ (Safety)')
    ax_logic.plot(df['time'], df['we'], color='#1f77b4', linewidth=2.5, label=r'$w_e$ (Efficiency)')
    ax_logic.fill_between(df['time'], df['ws'], alpha=0.1, color='#d62728')
    
    ax_twin.plot(df['time'], df['u_risk'], color='black', linestyle='--', linewidth=2, label='Risk $U$')
    
    ax_logic.set_title("Internal Weight & Risk Evolution", fontsize=16, fontweight='bold', pad=15)
    ax_logic.set_xlabel("Time [s]", fontsize=14, fontweight='bold')
    ax_logic.set_ylabel("CWN Contextual Weights", fontsize=14, fontweight='bold')
    ax_twin.set_ylabel("Physics Potential Energy", fontsize=14, fontweight='bold', rotation=270, labelpad=20)
    
    h1, l1 = ax_logic.get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    ax_logic.legend(h1+h2, l1+l2, loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/6_Real_Scenario_Study.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{SAVE_DIR}/6_Real_Scenario_Study.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f" ✅ 绘图成功！文件已存至 {SAVE_DIR}")

if __name__ == "__main__":
    # 执行
    try:
        data = record_real_scenario(num_steps=50)
        plot_figure_6_real(data)
    except Exception as e:
        print(f"❌ 运行失败，错误原因: {e}")