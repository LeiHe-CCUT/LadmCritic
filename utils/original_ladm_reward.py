import numpy as np
import math

class LadmReward:
    """
    根據 Ladm 論文思想實現的完整、有狀態的獎勵計算器。
    該類從 highway-env 的觀測和動作中提取物理量，計算
    行車安全能量 (DSE)、行車舒適能量 (DCE) 和行車效率能量 (DEE)，
    最終匯總為一個綜合的瞬時風險指標 D。
    獎勵 R_t 定義為 -D(s_{t+1})。
    """
    def __init__(self, dt=1/15):
        """
        初始化參數和狀態。
        :param dt: 模擬的時間步長 (秒)，用於計算加加速度。
                   highway-env 預設 policy_frequency=15Hz，所以 dt=1/15。
        """
        # 能量項的權重，與論文保持一致
        self.w_s = 0.7  # 安全權重
        self.w_c = 0.15 # 舒適權重
        self.w_e = 0.15 # 效率權重

        # 效率相關參數
        self.desired_speed = 30.0  # 期望速度 (m/s)，例如約 110 km/h
        self.max_speed = 40.0      # 可接受的最高速度 (m/s)

        # 舒適性計算所需的状态變數
        self.dt = dt
        self.last_acceleration = 0.0

    def reset(self):
        """
        在每個回合結束時重置狀態變數。
        """
        self.last_acceleration = 0.0

    def _calculate_dse(self, rel_distance_x, rel_velocity_x):
        """
        計算行車安全能量 (DSE)。
        基於前向碰撞時間 (Time-to-Collision, TTC) 的物理模型。
        僅在對方車輛速度更快時 (rel_velocity_x > 0) 才考慮 TTC。
        """
        # 只有當本車速度快於前車時 (相對速度為負)，才存在碰撞風險
        if rel_velocity_x < -1e-6:
            # TTC = 相對距離 / 相對速度 (取絕對值)
            ttc = rel_distance_x / -rel_velocity_x
            
            # 根據 TTC 計算 DSE，TTC 越小，風險能量越高
            # 這裡使用指數衰減函數，形式可以根據您的論文進行微調
            if ttc < 8.0: # 只考慮 8 秒內的碰撞風險
                dse = math.exp(-0.5 * (ttc - 1.0)) # 讓 TTC=1s 時風險最大
            else:
                dse = 0.0
        else:
            # 如果本車比前車慢或等速，則沒有前向碰撞風險
            dse = 0.0
        return dse

    def _calculate_dce(self, acceleration, jerk):
        """
        計算行車舒適能量 (DCE)。
        懲罰過大的縱向加速度和加加速度 (Jerk)。
        """
        # 舒適性能量是加速度和加加速度的平方和
        # 係數可以調整，以反映它們對舒適度的不同影響
        dce = 0.1 * acceleration**2 + 0.05 * jerk**2
        return dce

    def _calculate_dee(self, ego_speed):
        """
        計算行車效率能量 (DEE)。
        鼓勵車輛以期望速度行駛，並懲罰超速。
        """
        # 懲罰與期望速度的偏差
        speed_deviation = ego_speed - self.desired_speed
        dee_deviation = 0.2 * speed_deviation**2
        
        # 額外增加一個對超速的嚴厲懲罰 (非線性)
        dee_overspeed = 0.0
        if ego_speed > self.max_speed:
            dee_overspeed = 5.0 * (ego_speed - self.max_speed)**2
            
        return dee_deviation + dee_overspeed

    def _find_leading_vehicle(self, observation, lane_index):
        """
        從觀測數據中尋找正前方的車輛。
        :param observation: highway-env 的觀測陣列 (N_VEHICLES, FEATURES)
        :param lane_index: 自我車輛所在的車道索引
        :return: (相對距離, 相對速度) tuple。如果沒有前車，則返回 (None, None)。
        """
        min_rel_dist = float('inf')
        leading_vehicle_rel_vel = None

        ego_vehicle = observation[0]
        other_vehicles = observation[1:]

        for vehicle in other_vehicles:
            if vehicle[0] == 0: # 檢查車輛是否存在 (presence feature)
                continue
            
            # 檢查是否在同一車道
            # y 座標的差值可以判斷是否在同一車道，4m 是典型車道寬度
            if abs(vehicle[2] - ego_vehicle[2]) < 2.0: # y-position
                rel_dist_x = vehicle[1] - ego_vehicle[1] # x-position
                
                # 篩選出在前方且最近的車輛
                if 0 < rel_dist_x < min_rel_dist:
                    min_rel_dist = rel_dist_x
                    # 計算縱向相對速度
                    leading_vehicle_rel_vel = vehicle[3] - ego_vehicle[3] # vx
        
        if min_rel_dist == float('inf'):
            return None, None
        else:
            return min_rel_dist, leading_vehicle_rel_vel

    def compute_instantaneous_risk(self, observation, action, info):
        """
        計算給定狀態和動作下的瞬時總風險 D。
        :param observation: 環境返回的觀測 (numpy array)
        :param action: 智能體執行的動作 (numpy array)
        :param info: 環境返回的 info 字典
        :return: 總風險 D (純量)
        """
        # --- 1. 提取自我車輛物理資訊 ---
        ego_vehicle_obs = observation[0]
        ego_speed = np.linalg.norm(ego_vehicle_obs[3:5]) # 速度向量 [vx, vy] 的大小
        
        # 動作的第一個維度是縱向加速度
        current_acceleration = action[0] 
        
        # 計算加加速度 (Jerk)
        jerk = (current_acceleration - self.last_acceleration) / self.dt
        
        # 更新狀態以備下次計算
        self.last_acceleration = current_acceleration

        # --- 2. 計算各項能量 ---
        
        # (DSE) 安全能量
        # 從 info 中獲取車道資訊
        ego_lane_index = info.get("ego_lane", None)
        total_dse = 0.0
        if ego_lane_index is not None:
            rel_dist, rel_vel = self._find_leading_vehicle(observation, ego_lane_index)
            if rel_dist is not None:
                total_dse = self._calculate_dse(rel_dist, rel_vel)
        
        # (DCE) 舒適能量
        dce = self._calculate_dce(current_acceleration, jerk)
        
        # (DEE) 效率能量
        dee = self._calculate_dee(ego_speed)
                # --- 3. 加權匯總總風險 ---
        risk_d = self.w_s * total_dse + self.w_c * dce + self.w_e * dee
        
        # === 新增：對風險值進行縮放 ===
        # 將風險值縮小，讓獎勵訊號更平滑
        # 0.1 或 0.01 都是可以嘗試的起始值
        scaled_risk_d = 0.1 * risk_d 
        
        return scaled_risk_d # 返回縮放後的風險
        
