import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

# ================= 配置区域 =================
# 请在这里填入你 TensorBoard logs 的实际路径
# 格式: "显示名称": "文件夹路径"
log_dirs = {
    # 例子:
    # "SAC-MLP (Baseline)": "./logs/bc_finetune_experiment_1", 
    # "SAC-Ladm (Ours)": "./logs/sac_ladm_experiment_5",
    
    # 请修改下面为你真实的路径
    "SAC-MLP (Baseline)": "/data/users/HL/helei/LadmCritic/logs/mlp_baseline_experiment_6",
    "SAC-MLP + LadmReward": "/data/users/HL/helei/LadmCritic/logs/bc_finetune_experiment_5", 
    "LadmCritic (Ours)": "/data/users/HL/helei/LadmCritic/logs/ladm_experiment_11",
}

# 你想要提取的 TensorBoard Tag 名称 (根据你的代码 writer.add_scalar 的名称)
# 注意：Tag 名称必须与你训练代码中完全一致
tags_to_extract = {
    "Reward": "Baseline/BC_Finetune/Episode Reward", # 或者是 "Episode Reward"
    "Collision Rate": "Metrics/BC_Finetune/Collision Rate",
    "Success Rate": "Metrics/BC_Finetune/Success Rate",
    "Avg Speed": "Metrics/BC_Finetune/Average Speed (m/s)",
    "Avg Jerk": "Metrics/BC_Finetune/Average Jerk (m/s^3)",
    "Avg Accel": "Metrics/BC_Finetune/Average Acceleration (m/s^2)"
}

# 计算最后多少个 Episode 的平均值作为最终结果 (例如最后 100 个回合)
WINDOW_SIZE = 100 
# ===========================================

def get_tfevents_file(log_dir):
    """查找文件夹下最新的 tfevents 文件"""
    files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not files:
        return None
    # 返回修改时间最新的文件
    return max(files, key=os.path.getmtime)

def extract_data(name, log_dir):
    event_file = get_tfevents_file(log_dir)
    if not event_file:
        print(f"⚠️ 警告: 在 {name} 中找不到 tfevents 文件")
        return None

    # 加载 EventAccumulator
    ea = EventAccumulator(event_file)
    ea.Reload()

    results = {}
    
    print(f"正在处理: {name} ...")
    
    available_tags = ea.Tags()['scalars']

    for metric_name, tag_key in tags_to_extract.items():
        # 模糊匹配 Tag (防止代码稍微改动后 Tag 名字变了)
        # 如果找不到完全匹配，尝试部分匹配
        real_tag = tag_key
        if tag_key not in available_tags:
            candidates = [t for t in available_tags if metric_name in t or tag_key.split('/')[-1] in t]
            if candidates:
                real_tag = candidates[0]
            else:
                print(f"  - 找不到 Tag: {tag_key} (跳过)")
                results[metric_name] = ("N/A", "N/A")
                continue
        
        # 提取标量数据
        scalars = ea.Scalars(real_tag)
        values = [s.value for s in scalars]
        
        # 取最后 WINDOW_SIZE 个点的数据
        if len(values) > WINDOW_SIZE:
            recent_values = values[-WINDOW_SIZE:]
        else:
            recent_values = values
            
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        results[metric_name] = (mean_val, std_val)
        
    return results

def main():
    all_data = []
    
    for method_name, path in log_dirs.items():
        if "REPLACE" in path:
            continue # 跳过未配置的路径
            
        metrics = extract_data(method_name, path)
        if metrics:
            row = {"Method": method_name}
            for k, v in metrics.items():
                if v[0] == "N/A":
                    row[k] = "N/A"
                else:
                    # 格式化字符串: Mean ± Std
                    if "Rate" in k: # 如果是比率，显示百分比
                        row[k] = f"{v[0]*100:.1f} ± {v[1]*100:.1f}%"
                    else:
                        row[k] = f"{v[0]:.2f} ± {v[1]:.2f}"
            all_data.append(row)

    if not all_data:
        print("\n❌ 没有提取到数据，请检查 'log_dirs' 路径配置是否正确。")
        return

    # 创建 DataFrame 并打印 Markdown 表格
    df = pd.DataFrame(all_data)
    print("\n" + "="*50)
    print("✅ 请复制以下内容发送给 Gemini:")
    print("="*50 + "\n")
    print(df.to_markdown(index=False))
    print("\n" + "="*50)

if __name__ == "__main__":
    main()