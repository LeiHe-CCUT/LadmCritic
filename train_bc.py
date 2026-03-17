# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import numpy as np
# import os
# import yaml
# from tqdm import tqdm

# from models.actor import Actor # 從您的專案中匯入 Actor 模型
# from torch.utils.tensorboard import SummaryWriter

# class ExpertDataset(Dataset):
#     """
#     用於 PyTorch 的專家數據集類別。
#     """
#     def __init__(self, dataset_dir='./dataset'):
#         obs_path = os.path.join(dataset_dir, 'observations.npy')
#         act_path = os.path.join(dataset_dir, 'actions.npy')
        
#         try:
#             self.observations = np.load(obs_path)
#             self.actions = np.load(act_path)
#             print(f"從 '{dataset_dir}' 成功載入 {len(self.observations)} 筆專家數據。")
#         except FileNotFoundError:
#             print(f"錯誤：在 '{dataset_dir}' 中找不到數據集檔案。請先運行 generate_expert_data.py。")
#             self.observations = np.array([])
#             self.actions = np.array([])

#     def __len__(self):
#         return len(self.observations)

#     def __getitem__(self, idx):
#         return torch.FloatTensor(self.observations[idx]), torch.FloatTensor(self.actions[idx])

# def train_behavioral_cloning():
#     print("開始行為克隆 (BC) 預訓練...")
    
#     # --- 從 config 檔案載入超參數 ---
#     try:
#         # === 核心修正：指定檔案的編碼格式 ===
#         with open('configs/bc_config.yaml', 'r', encoding='utf-8') as f: # <-- 新增 encoding='utf-8'
#             config = yaml.safe_load(f)
#     except FileNotFoundError:
#         print("警告：找不到 'configs/bc_config.yaml'。將使用預設超參數。")
#         config = {
#             'epochs': 50,
#             'batch_size': 256,
#             'learning_rate': 1e-4,
#             'validation_split': 0.1
#         }

#     EPOCHS = config['epochs']
#     BATCH_SIZE = config['batch_size']
#     LEARNING_RATE = config['learning_rate']
#     VALIDATION_SPLIT = config['validation_split']
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ... 後續所有程式碼保持不變 ...
#     writer = SummaryWriter(log_dir="./logs/bc_training")
    
#     full_dataset = ExpertDataset()
#     if len(full_dataset) == 0:
#         return

#     val_size = int(len(full_dataset) * VALIDATION_SPLIT)
#     train_size = len(full_dataset) - val_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     state_dim = full_dataset.observations.shape[1]
#     action_dim = full_dataset.actions.shape[1]
#     max_action = 1.0

#     actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)

#     best_val_loss = float('inf')

#     for epoch in range(EPOCHS):
#         actor.train()
#         total_train_loss = 0
#         for states, expert_actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [訓練]"):
#             states, expert_actions = states.to(DEVICE), expert_actions.to(DEVICE)
#             action_means, _ = actor(states)
#             loss = criterion(action_means, expert_actions)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()
        
#         avg_train_loss = total_train_loss / len(train_loader)
#         writer.add_scalar('BC/Train Loss', avg_train_loss, epoch)

#         actor.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for states, expert_actions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [驗證]"):
#                 states, expert_actions = states.to(DEVICE), expert_actions.to(DEVICE)
#                 action_means, _ = actor(states)
#                 loss = criterion(action_means, expert_actions)
#                 total_val_loss += loss.item()
        
#         avg_val_loss = total_val_loss / len(val_loader)
#         writer.add_scalar('BC/Validation Loss', avg_val_loss, epoch)
        
#         print(f"Epoch {epoch+1}/{EPOCHS} | 訓練損失: {avg_train_loss:.6f} | 驗證損失: {avg_val_loss:.6f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             output_path = './trained_models/bc_actor_best.pth'
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             torch.save(actor.state_dict(), output_path)
#             print(f"*** 找到新的最佳模型，驗證損失: {best_val_loss:.6f}。已儲存至 '{output_path}' ***")

#     final_output_path = './trained_models/bc_actor_final.pth'
#     torch.save(actor.state_dict(), final_output_path)
#     writer.close()
    
#     print(f"\n行為克隆訓練完成！最終模型已儲存至: '{final_output_path}'")
#     print(f"表現最佳的模型已儲存至: './trained_models/bc_actor_best.pth'")

# if __name__ == '__main__':
#     train_behavioral_cloning()


# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import numpy as np
# import os
# import yaml
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# # --- 假設這是您的模型引入 ---
# # 如果您沒有 models 資料夾，請確保 Actor 類別定義在同一個檔案或正確的路徑
# from models.actor import Actor 

# class ExpertDataset(Dataset):
#     """
#     用於 PyTorch 的專家數據集類別。
#     """
#     def __init__(self, dataset_dir='./dataset'):
#         obs_path = os.path.join(dataset_dir, 'observations.npy')
#         act_path = os.path.join(dataset_dir, 'actions.npy')
        
#         try:
#             self.observations = np.load(obs_path)
#             self.actions = np.load(act_path)
#             print(f"從 '{dataset_dir}' 成功載入 {len(self.observations)} 筆專家數據。")
#         except FileNotFoundError:
#             print(f"錯誤：在 '{dataset_dir}' 中找不到數據集檔案。請先運行 generate_expert_data.py。")
#             self.observations = np.array([])
#             self.actions = np.array([])

#     def __len__(self):
#         return len(self.observations)

#     def __getitem__(self, idx):
#         return torch.FloatTensor(self.observations[idx]), torch.FloatTensor(self.actions[idx])

# def train_behavioral_cloning():
#     # 1. 檢測並設定裝置
#     if torch.cuda.is_available():
#         DEVICE = torch.device("cuda")
#         # 針對固定輸入大小的網絡加速
#         torch.backends.cudnn.benchmark = True
#         print(f"🔥 檢測到 GPU，將使用: {torch.cuda.get_device_name(0)}")
#     else:
#         DEVICE = torch.device("cpu")
#         print("⚠️ 未檢測到 GPU，將使用 CPU 進行訓練（速度較慢）。")

#     print("開始行為克隆 (BC) 預訓練...")
    
#     # --- 從 config 檔案載入超參數 ---
#     try:
#         with open('configs/bc_config.yaml', 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
#     except FileNotFoundError:
#         print("警告：找不到 'configs/bc_config.yaml'。將使用預設超參數。")
#         config = {
#             'epochs': 50,
#             'batch_size': 256,
#             'learning_rate': 1e-4,
#             'validation_split': 0.1
#         }

#     EPOCHS = config['epochs']
#     BATCH_SIZE = config['batch_size']
#     LEARNING_RATE = config['learning_rate']
#     VALIDATION_SPLIT = config['validation_split']

#     writer = SummaryWriter(log_dir="./logs/bc_training")
    
#     full_dataset = ExpertDataset()
#     if len(full_dataset) == 0:
#         return

#     val_size = int(len(full_dataset) * VALIDATION_SPLIT)
#     train_size = len(full_dataset) - val_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
#     # 2. 優化 DataLoader (加入 pin_memory)
#     # 如果使用 CUDA，開啟 pin_memory 可以加速數據從 CPU 複製到 GPU
#     loader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    
#     # 注意：如果 ExpertDataset 數據量非常大，可以考慮 num_workers > 0
#     # 但因為這裡是直接讀取 numpy array 到內存，num_workers=0 通常最穩定
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    
#     state_dim = full_dataset.observations.shape[1]
#     action_dim = full_dataset.actions.shape[1]
#     max_action = 1.0

#     # 3. 將模型移動到 GPU
#     actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)

#     best_val_loss = float('inf')

#     for epoch in range(EPOCHS):
#         actor.train()
#         total_train_loss = 0
        
#         # 訓練迴圈
#         with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [訓練]", unit="batch") as tepoch:
#             for states, expert_actions in tepoch:
#                 # 4. 將數據移動到 GPU (非阻塞傳輸以提升速度)
#                 states = states.to(DEVICE, non_blocking=True)
#                 expert_actions = expert_actions.to(DEVICE, non_blocking=True)
                
#                 action_means, _ = actor(states)
#                 loss = criterion(action_means, expert_actions)
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 total_train_loss += loss.item()
#                 tepoch.set_postfix(loss=loss.item()) # 即時顯示 loss
        
#         avg_train_loss = total_train_loss / len(train_loader)
#         writer.add_scalar('BC/Train Loss', avg_train_loss, epoch)

#         actor.eval()
#         total_val_loss = 0
        
#         # 驗證迴圈
#         with torch.no_grad():
#             for states, expert_actions in val_loader:
#                 states = states.to(DEVICE, non_blocking=True)
#                 expert_actions = expert_actions.to(DEVICE, non_blocking=True)
                
#                 action_means, _ = actor(states)
#                 loss = criterion(action_means, expert_actions)
#                 total_val_loss += loss.item()
        
#         avg_val_loss = total_val_loss / len(val_loader)
#         writer.add_scalar('BC/Validation Loss', avg_val_loss, epoch)
        
#         print(f"Epoch {epoch+1}/{EPOCHS} | 訓練損失: {avg_train_loss:.6f} | 驗證損失: {avg_val_loss:.6f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             output_path = './trained_models/bc_actor_best.pth'
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             torch.save(actor.state_dict(), output_path)
#             print(f"*** 找到新的最佳模型，驗證損失: {best_val_loss:.6f}。已儲存至 '{output_path}' ***")

#     final_output_path = './trained_models/bc_actor_final.pth'
#     torch.save(actor.state_dict(), final_output_path)
#     writer.close()
    
#     print(f"\n行為克隆訓練完成！最終模型已儲存至: '{final_output_path}'")
#     print(f"表現最佳的模型已儲存至: './trained_models/bc_actor_best.pth'")

# if __name__ == '__main__':
#     # 設置隨機種子以確保可重複性 (可選)
#     # torch.manual_seed(42)
#     # np.random.seed(42)
#     train_behavioral_cloning()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# 引入混合精度訓練模組
from torch.cuda.amp import autocast, GradScaler

# 假設這是您的模型引入
from models.actor import Actor 

class ExpertDataset(Dataset):
    def __init__(self, dataset_dir='./dataset'):
        obs_path = os.path.join(dataset_dir, 'observations.npy')
        act_path = os.path.join(dataset_dir, 'actions.npy')
        
        try:
            self.observations = np.load(obs_path)
            self.actions = np.load(act_path)
            print(f"從 '{dataset_dir}' 成功載入 {len(self.observations)} 筆專家數據。")
        except FileNotFoundError:
            print(f"錯誤：在 '{dataset_dir}' 中找不到數據集檔案。請先運行 generate_expert_data.py。")
            self.observations = np.array([])
            self.actions = np.array([])

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.observations[idx]), torch.FloatTensor(self.actions[idx])

def train_behavioral_cloning():
    # 1. 檢測並設定裝置
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"🔥 檢測到 GPU，將使用: {torch.cuda.get_device_name(0)} (已啟用 AMP 混合精度)")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️ 未檢測到 GPU，將使用 CPU 進行訓練（速度較慢）。")

    print("開始行為克隆 (BC) 預訓練...")
    
    # --- 載入設定 ---
    try:
        with open('configs/bc_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("警告：找不到設定檔，使用預設值。")
        config = {
            'epochs': 50,
            'batch_size': 256,
            'learning_rate': 3e-4, # 稍微調高一點初始 LR，配合 Scheduler
            'validation_split': 0.1,
            'weight_decay': 1e-4   # 新增權重衰減
        }

    EPOCHS = config.get('epochs', 50)
    BATCH_SIZE = config.get('batch_size', 256)
    LEARNING_RATE = config.get('learning_rate', 3e-4)
    VALIDATION_SPLIT = config.get('validation_split', 0.1)
    WEIGHT_DECAY = config.get('weight_decay', 1e-4) # 防止過擬合

    writer = SummaryWriter(log_dir="./logs/bc_training")
    
    full_dataset = ExpertDataset()
    if len(full_dataset) == 0: return

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 2. DataLoader 優化
    # num_workers=2 通常是對於記憶體內數據的一個甜蜜點，過高反而會有開銷
    loader_kwargs = {'pin_memory': True, 'num_workers': 2} if torch.cuda.is_available() else {}
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    
    state_dim = full_dataset.observations.shape[1]
    action_dim = full_dataset.actions.shape[1]
    max_action = 1.0

    # 模型初始化
    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    criterion = nn.MSELoss()
    
    # 3. 優化器加入 weight_decay (L2 正則化)
    optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 4. [新增] 學習率調度器 (Cosine Annealing)
    # 這會讓 LR 在訓練過程中像餘弦波一樣下降，通常能找到更優的解
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 5. [新增] 混合精度梯度縮放器
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        actor.train()
        total_train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [訓練]", unit="batch") as tepoch:
            for states, expert_actions in tepoch:
                states = states.to(DEVICE, non_blocking=True)
                expert_actions = expert_actions.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                
                # --- 混合精度前向傳播 ---
                if scaler:
                    with autocast():
                        action_means, _ = actor(states) # 我們只監督 Mean，忽略 Log_std
                        loss = criterion(action_means, expert_actions)
                    
                    # 混合精度反向傳播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 傳統 FP32 模式
                    action_means, _ = actor(states)
                    loss = criterion(action_means, expert_actions)
                    loss.backward()
                    optimizer.step()
                # -----------------------

                total_train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # 更新學習率
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('BC/Train Loss', avg_train_loss, epoch)
        writer.add_scalar('BC/Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # --- 驗證 ---
        actor.eval()
        total_val_loss = 0
        with torch.no_grad():
            for states, expert_actions in val_loader:
                states = states.to(DEVICE, non_blocking=True)
                expert_actions = expert_actions.to(DEVICE, non_blocking=True)
                
                # 驗證時也可以開 autocast 加速推論
                if torch.cuda.is_available():
                    with autocast():
                        action_means, _ = actor(states)
                        loss = criterion(action_means, expert_actions)
                else:
                    action_means, _ = actor(states)
                    loss = criterion(action_means, expert_actions)
                    
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('BC/Validation Loss', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_path = './trained_models/bc_actor_best.pth'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(actor.state_dict(), output_path)
            print(f"*** 新的最佳模型 (Val Loss: {best_val_loss:.6f}) ***")

    final_output_path = './trained_models/bc_actor_final.pth'
    torch.save(actor.state_dict(), final_output_path)
    writer.close()
    
    print(f"\n訓練完成！最佳模型: ./trained_models/bc_actor_best.pth")

if __name__ == '__main__':
    train_behavioral_cloning()