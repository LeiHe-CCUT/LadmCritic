import numpy as np

class RuleBasedAgent:
    """
    規則型智能體的基礎類別。
    """
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        """
        根據觀測來決定動作。子類別必須實作這個方法。
        """
        raise NotImplementedError()

class AggressiveAgent(RuleBasedAgent):
    """
    一個魯莽的、不安全的駕駛智能體。
    它的目標是：盡可能快地開車，並保持一個極短的跟車距離。
    """
    # 跟車距離閾值 (單位：公尺)
    TOO_CLOSE_THRESHOLD = 8  # 極度危險的距離
    CLOSE_THRESHOLD = 15     # 依然很近的距離

    # 速度控制
    TARGET_SPEED = 35.0  # 期望達到的高速 (m/s)

    def act(self, observation):
        """
        根據前方車輛的距離和自身速度來決定動作。
        """
        # observation[0] 是自我車輛, observation[1] 是正前方車輛
        # 每個車輛的特徵是 ["presence", "x", "y", "vx", "vy", ...]
        
        front_vehicle = observation[1]
        has_front_vehicle = front_vehicle[0] == 1.0
        
        ego_speed = np.linalg.norm(observation[0][3:5]) # 自我車輛速度

        # 預設動作：全油門，不轉向
        acceleration = 1.0
        steering = 0.0

        if has_front_vehicle:
            # front_vehicle[1] 是相對 x 距離, front_vehicle[2] 是相對 y 距離
            distance_to_front = np.linalg.norm(front_vehicle[1:3])

            if distance_to_front < self.TOO_CLOSE_THRESHOLD:
                # 距離過近，緊急剎車
                acceleration = -1.0
            elif distance_to_front < self.CLOSE_THRESHOLD:
                # 距離較近，稍微減速以維持極短的跟車距離
                acceleration = -0.5
            # 如果距離大於 CLOSE_THRESHOLD，則繼續全油門追趕
        
        # 速度限制：如果已超速，則不再加速
        if ego_speed > self.TARGET_SPEED:
            acceleration = 0.0

        # 返回連續動作 [加速度, 轉向]
        return np.array([acceleration, steering])
