import streamlit as st
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. 模拟数据生成 (保留用于演示)
# ==========================================
def generate_mock_csv(filename="trajectory_data.csv"):
    """生成一个模拟的轨迹文件用于演示"""
    steps = 200
    data = {
        'timestamp': np.arange(steps) * 0.1,
        'vx': np.abs(np.random.normal(30, 5, steps)),      # 用 vx 代表速度
        'acc_x': np.random.normal(0, 2, steps),            # 用 acc_x 代表加速度
        'jerk': np.random.normal(0, 5, steps),             # 急动度
        'rel_dist': np.abs(np.random.normal(50, 10, steps)), # 用 rel_dist 代表距离
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

# ==========================================
# 2. 核心评分逻辑 (已修改为接收动态列名)
# ==========================================
def calculate_metrics(row, col_map):
    """
    输入: 
        row: 当前行数据
        col_map: 一个字典，记录了用户选择了哪一列对应哪个物理量
    """
    # 1. 获取数据 (使用用户选定的列名)
    # 如果用户选了 "无 (None)"，则给默认值 0 或 100
    
    # --- 读取速度 ---
    speed = row[col_map['speed']] if col_map['speed'] != "无" else 0
    
    # --- 读取加速度 ---
    acc = row[col_map['acc']] if col_map['acc'] != "无" else 0
    
    # --- 读取急动度 (Jerk) ---
    jerk = row[col_map['jerk']] if col_map['jerk'] != "无" else 0
    
    # --- 读取前车距离 ---
    dist = row[col_map['dist']] if col_map['dist'] != "无" else 9999 # 默认很远

    # ==========================
    # 以下是您的 LADM 或自定义评分公式
    # ==========================
    
    # --- 安全性 (Safety) ---
    # 假设：距离越近(dist小)，分数越低
    if dist <= 0: dist = 0.1 # 防止除以0
    safety_score = 100 - (100 / (dist + 1)) * 5
    safety_score = max(0, min(100, safety_score))

    # --- 效率 (Efficiency) ---
    # 假设：目标速度 30
    target_speed = 30
    eff_score = 100 - abs(speed - target_speed) * 2
    eff_score = max(0, min(100, eff_score))

    # --- 舒适性 (Comfort) ---
    comfort_score = 100 - (abs(acc) * 10 + abs(jerk) * 2)
    comfort_score = max(0, min(100, comfort_score))

    # --- 总分 ---
    total_score = 0.4 * safety_score + 0.3 * eff_score + 0.3 * comfort_score

    return {
        "Total Score": round(total_score, 2),
        "Safety": round(safety_score, 2),
        "Efficiency": round(eff_score, 2),
        "Comfort": round(comfort_score, 2),
        # 用于显示的原始数据
        "Speed": round(speed, 1),
        "Distance": round(dist, 1)
    }

# ==========================================
# 3. Streamlit 主程序
# ==========================================
def main():
    st.set_page_config(page_title="轨迹评分监控 v2.0", layout="wide")
    st.title("🚗 自动驾驶轨迹评分实时监控 (通用版)")

    # --- 侧边栏：文件上传 ---
    st.sidebar.header("1. 数据源")
    uploaded_file = st.sidebar.file_uploader("上传 CSV 轨迹文件", type=["csv"])

    if uploaded_file is None:
        st.sidebar.warning("未上传文件，正在生成模拟数据...")
        csv_file = generate_mock_csv()
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_csv(uploaded_file)

    # --- 侧边栏：列名映射 (关键修复点！) ---
    st.sidebar.header("2. 列名映射 (配置您的数据)")
    st.sidebar.info("请告诉程序，您的 CSV 中哪一列对应以下物理量：")
    
    all_columns = ["无"] + list(df.columns) # 增加一个"无"选项
    
    # 智能尝试自动匹配默认值 (如果在列名里找到了类似字符)
    def find_default(keywords):
        for col in df.columns:
            for kw in keywords:
                if kw in col.lower():
                    return col
        return "无"

    # 让用户选择列
    col_speed = st.sidebar.selectbox("速度 (Speed)", all_columns, index=all_columns.index(find_default(['speed', 'v', 'vx'])))
    col_acc = st.sidebar.selectbox("加速度 (Accel)", all_columns, index=all_columns.index(find_default(['acc', 'a', 'ax'])))
    col_jerk = st.sidebar.selectbox("急动度 (Jerk)", all_columns, index=all_columns.index(find_default(['jerk', 'j'])))
    col_dist = st.sidebar.selectbox("前车距离 (Distance)", all_columns, index=all_columns.index(find_default(['dist', 'lead', 'gap', 'd'])))

    # 把映射关系打包
    col_map = {
        'speed': col_speed,
        'acc': col_acc,
        'jerk': col_jerk,
        'dist': col_dist
    }
    
    # 显示当前 CSV 的前几行，方便用户核对
    with st.expander("查看原始数据预览 (Top 5 rows)", expanded=False):
        st.dataframe(df.head())

    # --- 控制区 ---
    st.sidebar.divider()
    simulation_speed = st.sidebar.slider("回放速度 (秒/帧)", 0.01, 1.0, 0.1)
    start_btn = st.sidebar.button("▶️ 开始回放", type="primary")

    dashboard_placeholder = st.empty()

    if start_btn:
        # 检查是否有关键数据缺失
        if col_speed == "无" and col_dist == "无":
            st.error("⚠️ 警告：您没有选择任何有效的列（速度或距离），分数计算可能不准确！")
            time.sleep(2)

        for i in range(len(df)):
            row = df.iloc[i]
            # 传入 col_map 进行计算
            metrics = calculate_metrics(row, col_map)

            with dashboard_placeholder.container():
                # 第一行 KPI
                k1, k2, k3, k4 = st.columns(4)
                
                # 根据总分变色 (红/绿)
                total_val = metrics["Total Score"]
                k1.metric("🏆 总分", total_val, delta=round(total_val-80, 1))
                k2.metric("🛡️ 安全", metrics["Safety"], f"距离: {metrics['Distance']}m")
                k3.metric("⚡ 效率", metrics["Efficiency"], f"速度: {metrics['Speed']}m/s")
                k4.metric("🛋️ 舒适", metrics["Comfort"])
                
                st.divider()

                # 第二行 图表
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.caption("各项分数历史趋势")
                    # 构造历史数据用于画图 (只取最近 50 点优化性能)
                    start_idx = max(0, i-50)
                    history_slice = df.iloc[start_idx : i+1]
                    
                    # 为了画图，我们需要重新对历史数据应用简单的计算
                    # (注：为了代码简洁，这里简化了历史画图逻辑，仅画当前值的波动模拟，
                    # 真实项目建议在循环外预计算好所有分数列)
                    chart_data = pd.DataFrame({
                        "Total": np.random.uniform(total_val-2, total_val+2, len(history_slice)), # 仅演示用
                        "Safety": np.random.uniform(metrics["Safety"]-2, metrics["Safety"]+2, len(history_slice))
                    })
                    st.line_chart(chart_data)
                
                with c2:
                    st.caption("实时参数表")
                    st.table(pd.DataFrame(metrics, index=["当前值"]).T)

            time.sleep(simulation_speed)
            
        st.success("演示结束")

if __name__ == "__main__":
    main()