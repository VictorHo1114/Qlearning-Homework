import numpy as np
import random
from tqdm import tqdm

# ==========================================
# 1. 參數設定區
# ==========================================
n_episodes = 10000       # 總回合數
n_steps = 100            # 每回合步數
gamma = 0.9              # 折扣因子

# --- 實驗變數：全部固定 ---
# 這是造成階梯狀與震盪的主因，但我們用 Patience 來解決停止問題
alpha = 0.1              # 固定學習率
epsilon = 0.1            # 固定探索率

# --- 經濟參數 (保持一致) ---
beta = 10
eps_A = 0.5
eps_p = -1.5
c = 2
penalty = -1_000_000

# --- 搜尋範圍與網格 ---
A_min, A_max = 0.0, 10.0
p_min, p_max = 0.1, 20.0 
n_A = 101 
n_p = 200 

A_values = np.linspace(A_min, A_max, n_A)
p_values = np.linspace(p_min, p_max, n_p)

# 動作集合
actions = [(-1, -1), (-1, 0), (-1, 1),
           ( 0, -1), ( 0, 0), ( 0, 1),
           ( 1, -1), ( 1, 0), ( 1, 1)]
num_actions = len(actions)

# Q-table 初始化
Q_table = np.zeros((n_A, n_p, num_actions))

# ==========================================
# 2. [關鍵修改] 終止條件設計: Patience
# ==========================================
# 對於 Fixed 參數，我們不看 delta Q (因為它永遠在跳)，我們只看 "有沒有進步"
patience_limit = 5000         # 忍耐極限：允許連續 3000 回合沒有進步
min_episodes_run = 3000       # 最少跑幾回合 (避免一開始運氣好就停)
episodes_without_improvement = 0 # 計數器

global_best_profit = -np.inf  # 紀錄歷史以來發現過的 "最佳策略利潤"

# 數據紀錄
history_profit = []
history_delta_q = []
history_best_A = []
history_best_p = []

def get_reward(A, p):
    if A < 0 or p <= 0:
        return penalty
    Q = beta * (A ** eps_A) * (p ** eps_p)
    pi = (p - c) * Q - A
    return pi

# ==========================================
# 3. 主訓練迴圈
# ==========================================
progress_bar = tqdm(range(n_episodes), desc="Training [Fixed + Patience]")

for episode in progress_bar:
    
    # 隨機起始點
    A_idx = random.randint(0, n_A - 1)
    p_idx = random.randint(0, n_p - 1)

    delta_sum = 0.0
    update_count = 0

    for step in range(n_steps):
        # Epsilon-Greedy (固定 epsilon)
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, num_actions - 1)
        else:
            action_index = np.argmax(Q_table[A_idx, p_idx])

        dA, dp = actions[action_index]
        new_A_idx = max(0, min(n_A - 1, A_idx + dA))
        new_p_idx = max(0, min(n_p - 1, p_idx + dp))
        
        A = A_values[new_A_idx]
        p = p_values[new_p_idx]
        reward = get_reward(A, p)

        # Q-Learning 更新 (固定 Alpha)
        old_q = Q_table[A_idx, p_idx, action_index]
        best_next = np.max(Q_table[new_A_idx, new_p_idx])
        
        td_target = reward + gamma * best_next
        Q_table[A_idx, p_idx, action_index] += alpha * (td_target - old_q)
        
        delta_sum += abs(Q_table[A_idx, p_idx, action_index] - old_q)
        update_count += 1
        
        A_idx, p_idx = new_A_idx, new_p_idx

    # === 數據記錄與 Patience 邏輯 ===
    avg_delta_q = delta_sum / max(1, update_count)

    # 1. 找出當前 Q-table 認為最好的策略 (Greedy Policy)
    best_idx_flat = np.argmax(np.max(Q_table, axis=2))
    best_A_idx, best_p_idx = np.unravel_index(best_idx_flat, (n_A, n_p))
    
    current_A = A_values[best_A_idx]
    current_p = p_values[best_p_idx]
    current_policy_profit = get_reward(current_A, current_p)
    
    # 存入歷史
    history_profit.append(current_policy_profit)
    history_delta_q.append(avg_delta_q)
    history_best_A.append(current_A)
    history_best_p.append(current_p)

    # 2. Patience 核心判斷
    # 如果找到了比 "歷史紀錄" 還要好的利潤 (必須有實質提升，大於極小值)
    if current_policy_profit > global_best_profit + 1e-5:
        global_best_profit = current_policy_profit
        episodes_without_improvement = 0  # 發現新大陸！重置耐心
        # tqdm.write(f"Ep {episode}: 發現新策略 π={global_best_profit:.4f} (重置耐心)")
    else:
        episodes_without_improvement += 1 # 沒進步，耐心 -1

    # 更新進度條
    progress_bar.set_description(
        f"π={current_policy_profit:.2f}, Patience={episodes_without_improvement}/{patience_limit}"
    )

    # 3. 檢查是否觸發停止
    if episode > min_episodes_run and episodes_without_improvement >= patience_limit:
        print(f"\n" + "="*40)
        print(f"*** 收斂達成 (Patience Exhausted) ***")
        print(f"停止回合: {episode}")
        print(f"原因: 連續 {patience_limit} 回合沒有找到更好的策略")
        print(f"="*40)
        break

print(f"最終結果: A={history_best_A[-1]:.3f}, p={history_best_p[-1]:.3f}, π={history_profit[-1]:.4f}")

# ==========================================
# 4. 視覺化繪圖區
# ==========================================
import matplotlib.pyplot as plt

print("正在繪製圖表...")
plt.figure(figsize=(15, 10))

# 圖 1: 利潤 (可以看到階梯狀，以及最後平盤多久後停止)
plt.subplot(2, 2, 1)
plt.plot(history_profit, label='Profit', color='blue')
plt.title(f'Profit Convergence (Best: {global_best_profit:.4f})')
plt.xlabel('Episode')
plt.ylabel('Profit')
plt.grid(True)

# 圖 2: Q值變動 (即便停止了，這張圖應該還是在震盪，證明了為什麼不能用 Delta Q 停)
plt.subplot(2, 2, 2)
plt.plot(history_delta_q, label='Avg ΔQ', color='orange', alpha=0.7, linewidth=0.5)
plt.yscale('log') 
plt.title('Delta Q (Oscillating due to fixed params)')
plt.xlabel('Episode')
plt.ylabel('Avg |Q_new - Q_old| (Log Scale)')
plt.grid(True)

# 圖 3: A 軌跡
plt.subplot(2, 2, 3)
plt.plot(history_best_A, label='Best A', color='green')
plt.title(f'Trajectory of A (Final: {history_best_A[-1]:.3f})')
plt.xlabel('Episode')
plt.grid(True)

# 圖 4: p 軌跡
plt.subplot(2, 2, 4)
plt.plot(history_best_p, label='Best p', color='red')
plt.title(f'Trajectory of p (Final: {history_best_p[-1]:.3f})')
plt.xlabel('Episode')
plt.grid(True)

plt.tight_layout()
plt.show()

