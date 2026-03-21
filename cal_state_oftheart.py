import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 第一部分：核心评估函数（用于您的真实实验数据）
# ==========================================
def evaluate_model_performance(y_true_human, y_pred_model, safe_scenario_indices, cut_in_scenario_indices):
    """
    计算评估器的三大核心指标：PCC, MAE, Hazard Sensitivity
    
    参数:
    y_true_human: 人类专家打分 (Ground Truth), 形状 (N,)
    y_pred_model: 评估器给出的打分, 形状 (N,)
    safe_scenario_indices: 安全巡航场景在数组中的索引列表
    cut_in_scenario_indices: 危险加塞场景在数组中的索引列表
    """
    # 1. Pearson Correlation Coefficient (PCC)
    pcc, _ = pearsonr(y_true_human, y_pred_model)
    
    # 2. Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true_human, y_pred_model)
    
    # 3. Hazard Sensitivity (Delta drop)
    mean_safe_score = np.mean(y_pred_model[safe_scenario_indices])
    mean_hazard_score = np.mean(y_pred_model[cut_in_scenario_indices])
    
    # 计算下降的百分比
    hazard_sensitivity = ((mean_safe_score - mean_hazard_score) / mean_safe_score) * 100
    
    return pcc, mae, hazard_sensitivity


# ==========================================
# 第二部分：论文数据生成与自动绘图（复现您的表格）
# ==========================================
def generate_paper_results_and_plot():
    # 录入论文中的对比数据
    data = {
        "Evaluator Framework": [
            "Onsite (Rule)", 
            "S2O (AAP 2025)", 
            "RTCE (ICRA 2025)", 
            "Offline Metric (IROS 2025)", 
            "LadmCritic (Ours)"
        ],
        "PCC": [0.654, 0.915, 0.890, 0.922, 0.942],
        "MAE": [11.20, 4.58, 5.12, 4.85, 4.12],
        "Hazard Sensitivity (%)": [1.80, 15.60, 14.80, 18.20, 28.50]
    }
    
    df = pd.DataFrame(data)
    
    # 打印格式化的表格输出
    print("================================================================")
    print("Table 5: Quantitative performance comparison with 2025 SOTA")
    print("================================================================")
    print(df.to_string(index=False))
    print("================================================================\n")
    
    # 绘制高颜值的论文对比图
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 定义颜色：基线模型用淡蓝色系，LadmCritic 用醒目的红色/橙色
    palette = ['#8da0cb', '#66c2a5', '#a6d854', '#ffd92f', '#e7298a']
    
    # 图 1: PCC (越高越好)
    sns.barplot(x="PCC", y="Evaluator Framework", data=df, ax=axes[0], palette=palette)
    axes[0].set_title("Pearson Correlation (PCC) ↑", fontweight='bold')
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_ylabel("")
    
    # 图 2: MAE (越低越好)
    sns.barplot(x="MAE", y="Evaluator Framework", data=df, ax=axes[1], palette=palette)
    axes[1].set_title("Mean Absolute Error (MAE) ↓", fontweight='bold')
    axes[1].set_xlim(0, 12.5)
    axes[1].set_ylabel("")
    axes[1].set_yticks([]) # 隐藏中间图的 y 轴标签以保持整洁
    
    # 图 3: Hazard Sensitivity (越高越好)
    sns.barplot(x="Hazard Sensitivity (%)", y="Evaluator Framework", data=df, ax=axes[2], palette=palette)
    axes[2].set_title("Hazard Sensitivity (Δdrop) ↑", fontweight='bold')
    axes[2].set_xlim(0, 32)
    axes[2].set_ylabel("")
    axes[2].set_yticks([])
    
    # 为柱状图添加数值标签
    for ax, metric in zip(axes, ["PCC", "MAE", "Hazard Sensitivity (%)"]):
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            label_text = f"{width:.3f}" if metric == "PCC" else f"{width:.2f}"
            if metric == "Hazard Sensitivity (%)": label_text += "%"
            ax.text(width + (0.01 if metric=="PCC" else 0.3), 
                    p.get_y() + p.get_height() / 2, 
                    label_text, 
                    ha="left", va="center", fontweight='bold')

    plt.tight_layout()
    plt.savefig("evaluation_metrics_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Visualization saved successfully as 'evaluation_metrics_comparison.png'")
    plt.show()

if __name__ == "__main__":
    # 执行生成论文图表
    generate_paper_results_and_plot()
    
    # ==== 如何在您的真实项目中使用核心函数的示例 ====
    # print("\n[Example] Running metric calculation on mock experimental arrays...")
    # np.random.seed(42)
    # y_human = np.random.normal(85, 10, 100)
    # y_model = y_human + np.random.normal(0, 4, 100) # 模拟一个较好的模型
    # safe_idx = np.arange(0, 50)
    # cutin_idx = np.arange(50, 100)
    # y_model[cutin_idx] -= 25 # 模拟面对加塞时扣分
    # 
    # pcc, mae, sensitivity = evaluate_model_performance(y_human, y_model, safe_idx, cutin_idx)
    # print(f"Mock LadmCritic - PCC: {pcc:.3f}, MAE: {mae:.2f}, Sensitivity: {sensitivity:.2f}%")