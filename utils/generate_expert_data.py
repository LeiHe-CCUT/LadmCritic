import gymnasium as gym
import highway_env
import numpy as np
from tqdm import tqdm
import os

class RobustExpertAgent:
    """
    一個更穩健、完全基於觀測數據進行決策的規則型專家。
    """
    def __init__(self, target_speed=30.0, safe_time_headway=1.5, max_acceleration=1.0, max_deceleration=-1.5):
        # --- 專家策略的超參數 ---
        self.target_speed = target_speed
        self.safe_time_headway = safe_time_headway  # 安全時距 (秒)
        self.max_acceleration = max_acceleration    # 最大油門
        self.max_deceleration = max_deceleration    # 最大剎車
        
        # 橫向控制 (Lane Keeping) 的 PID 參數
        self.pid_kp = 2.5  # 比例
        self.pid_kd = 0.5  # 微分
        self.pid_ki = 0.1  # 積分
        self.last_error = 0.0
        self.integral_error = 0.0

    def reset(self):
        """重置 PID 控制器的狀態"""
        self.last_error = 0.0
        self.integral_error = 0.0

    def act(self, obs):
        """
        根據觀測決定動作。
        obs: 未扁平化的觀測數據, shape (vehicles, features)
             features: ["presence", "x", "y", "vx", "vy"] (相對值)
        """
        # --- 橫向控制 (轉向) ---
        ego_obs = obs[0]
        # obs[0, 2] 是橫向偏移量 (y)
        lateral_error = ego_obs[2]
        
        # 使用 PID 控制器計算轉向
        derivative_error = lateral_error - self.last_error
        self.integral_error += lateral_error
        steering = - (self.pid_kp * lateral_error + 
                      self.pid_kd * derivative_error + 
                      self.pid_ki * self.integral_error)
        self.last_error = lateral_error

        # --- 縱向控制 (油門/剎車) ---
        ego_speed = np.linalg.norm(ego_obs[3:5]) # 自車速度
        
        # 預設：以目標速度巡航 (IDM 自由道路項)
        acceleration = self.max_acceleration * (1 - (ego_speed / self.target_speed)**4)

        # 跟車邏輯 (IDM 交互項)
        front_vehicle_obs = obs[1]
        if front_vehicle_obs[0] == 1.0: # 如果前方有車
            rel_dist_x = front_vehicle_obs[1]
            rel_vel_x = front_vehicle_obs[3]

            # 期望的最小跟車距離
            min_gap = 10.0 # 最小靜止距離
            desired_gap = min_gap + ego_speed * self.safe_time_headway
            
            # 如果實際距離小於期望距離，則需要減速
            if rel_dist_x < desired_gap:
                # 根據與安全距離的差距和相對速度來計算減速度
                gap_ratio = (desired_gap / rel_dist_x)**2
                interaction_term = -self.max_acceleration * gap_ratio
                acceleration += interaction_term
        
        # 將最終動作限制在 [-1, 1] 範圍內
        final_acceleration = np.clip(acceleration, self.max_deceleration / 5.0, self.max_acceleration / 5.0)
        final_steering = np.clip(steering, -1.0, 1.0)
        
        return np.array([final_acceleration, final_steering])


def generate_data(num_steps=50000):
    print(f"開始生成 {num_steps} 步的專家數據 (使用穩健版專家)...")

    # 使用與訓練時相同的環境設定
    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False, # 確保使用相對特徵
        },
        "action": {"type": "ContinuousAction"},
        "policy_frequency": 15,
        "vehicles_count": 20, # 增加交通密度以產生更多樣的數據
    }
    env = gym.make("highway-v0", config=env_config)
    
    expert = RobustExpertAgent()
    
    observations, actions = [], []
    obs, info = env.reset()
    expert.reset() # 重置專家內部狀態
    
    for _ in tqdm(range(num_steps), desc="生成數據"):
        action = expert.act(obs)
        observations.append(obs.flatten())
        actions.append(action)
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, info = env.reset()
            expert.reset() # 重置專家內部狀態
            
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    
    # === 新增：使用絕對路徑，避免執行位置問題 ===
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'observations.npy'), observations)
    np.save(os.path.join(output_dir, 'actions.npy'), actions)
    
    print(f"\n數據生成完畢！共 {len(observations)} 筆數據已儲存至 '{output_dir}' 資料夾。")
    env.close()

if __name__ == '__main__':
    # 執行前，先將此檔案移回專案根目錄，或確保 import 路徑正確
    generate_data()