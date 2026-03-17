import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.signal import savgol_filter

# ================= 配置: 你的真实路径 =================
log_dirs = {
    "SAC-MLP": r"D:\电脑文件\服务器搭建代码\helei\LadmCritic\logs\mlp_baseline_experiment_7",
    "Ablation": r"D:\电脑文件\服务器搭建代码\helei\LadmCritic\logs\bc_finetune_experiment_5", 
    "LadmCritic_Ours": r"D:\电脑文件\服务器搭建代码\helei\LadmCritic\logs\ladm_experiment_12",
}
target_tag_keyword = "Reward" 
save_dir = "./paper_data_analysis"
os.makedirs(save_dir, exist_ok=True)
# ====================================================

def get_tfevents_file(log_dir):
    files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    return max(files, key=os.path.getmtime) if files else None

def read_tb_data(log_dir, tag):
    event_file = get_tfevents_file(log_dir)
    if not event_file: return [], []
    
    # 扩大加载限制，确保读取所有点
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    
    avail_tags = ea.Tags()['scalars']
    real_tag = next((t for t in avail_tags if tag in t or tag.split('/')[-1] in t), None)
    
    if not real_tag: return [], []
    events = ea.Scalars(real_tag)
    return [e.step for e in events], [e.value for e in events]

# 学术级平滑算法 1: EMA (Tensorboard默认)
def smooth_ema(scalars, weight=0.95):  # weight 越高越平滑 (比如0.95-0.99)
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 学术级平滑算法 2: 窗口滑动平均 (均值滤波)
def smooth_moving_average(scalars, window_size=50):
    return pd.Series(scalars).rolling(window=window_size, min_periods=1).mean().tolist()

print("正在提取并处理数据...")
plt.figure(figsize=(10, 6))

all_data = {}

for name, path in log_dirs.items():
    steps, values = read_tb_data(path, target_tag_keyword)
    if not steps:
        print(f"❌ 找不到 {name} 的数据")
        continue
        
    # 保存原始数据到 DataFrame
    df = pd.DataFrame({'Step': steps, 'Raw_Reward': values})
    
    # 应用多种平滑算法对比
    df['EMA_0.95'] = smooth_ema(values, weight=0.95)
    df['Moving_Avg_50'] = smooth_moving_average(values, window_size=50)
    
    # 保存到 CSV 供你检查
    csv_path = os.path.join(save_dir, f"{name}_raw_and_smoothed.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ {name} 数据已提取并保存至: {csv_path} (共 {len(steps)} 个数据点)")
    
    # 在图上绘制最稳的曲线 (这里我们用 EMA 0.95 作为主线展示)
    plt.plot(df['Step'], df['EMA_0.95'], label=f"{name} (Smoothed)", linewidth=2)
    # 降低透明度画出原始数据的阴影，这是顶级期刊极其喜欢的画法！
    plt.plot(df['Step'], df['Raw_Reward'], alpha=0.15, color=plt.gca().lines[-1].get_color())

plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Episode Reward', fontsize=12)
plt.title('Training Convergence (with Authentic Variance Shadow)', fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig(os.path.join(save_dir, "authentic_learning_curve.png"), dpi=300)
print(f"✅ 科学合规的对比图已生成。")