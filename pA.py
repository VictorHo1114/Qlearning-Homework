import numpy as np
import random
from tqdm import tqdm

# ==========================================
# 1. 核心參數 (Control Group: Fixed)
# ==========================================
n_episodes = 15000
n_steps = 100
gamma = 0.9

# --- 實驗變數: 全部固定 ---
alpha = 0.1        # 學習率固定
epsilon = 0.1      # 探索率固定

# --- 經濟與環境參數 (完全一致) ---
beta = 10
eps_A = 0.5
eps_p = -1.5
c = 2
penalty = -1_000_000

A_min, A_max = 0.0, 10.0
p_min, p_max = 0.1, 20.0 
n_A = 101 
n_p = 200 

A_values = np.linspace(A_min, A_max, n_A)
p_values = np.linspace(p_min, p_max, n_p)

actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
num_actions = len(actions)

Q_table = np.zeros((n_A, n_p, num_actions))

# --- 終止條件 (完全一致) ---
# 只有當 "利潤不變" 且 "Q值變動極小" 時才停止
eps_pi_threshold = 1e-4      
stop_delta_q_threshold = 1e-2 # 設定為 0.01
stable_episodes_req = 30     
min_episodes_run = 3000      

# 數據紀錄
history_profit = []
history_delta_q = []
history_best_A = []
history_best_p = []

def get_reward(A, p):
    if A < 0 or p <= 0: return penalty
    Q = beta * (A ** eps_A) * (p ** eps_p)
    pi = (p - c) * Q - A
    return pi

# ==========================================
# 2. 訓練迴圈
# ==========================================
stable_count = 0
prev_best_profit = -np.inf

progress_bar = tqdm(range(n_episodes), desc="[Fixed] Training")

for episode in progress_bar:
    
    A_idx = random.randint(0, n_A - 1)
    p_idx = random.randint(0, n_p - 1)
    
    delta_sum = 0.0
    update_count = 0
    
    for step in range(n_steps):
        # 策略: 固定 Epsilon
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, num_actions - 1)
        else:
            action_index = np.argmax(Q_table[A_idx, p_idx])
            
        dA, dp = actions[action_index]
        new_A_idx = max(0, min(n_A - 1, A_idx + dA))
        new_p_idx = max(0, min(n_p - 1, p_idx + dp))
        
        reward = get_reward(A_values[new_A_idx], p_values[new_p_idx])
        
        # 更新: 固定 Alpha
        old_q = Q_table[A_idx, p_idx, action_index]
        best_next = np.max(Q_table[new_A_idx, new_p_idx])
        td_target = reward + gamma * best_next
        
        Q_table[A_idx, p_idx, action_index] += alpha * (td_target - old_q)
        
        delta_sum += abs(Q_table[A_idx, p_idx, action_index] - old_q)
        update_count += 1
        A_idx, p_idx = new_A_idx, new_p_idx
        
    # --- 紀錄與收斂檢查 ---
    avg_delta_q = delta_sum / max(1, update_count)
    
    # 找當前最佳解
    best_idx_flat = np.argmax(np.max(Q_table, axis=2))
    best_A_i, best_p_i = np.unravel_index(best_idx_flat, (n_A, n_p))
    pi_best = get_reward(A_values[best_A_i], p_values[best_p_i])
    
    history_profit.append(pi_best)
    history_delta_q.append(avg_delta_q)
    history_best_A.append(A_values[best_A_i])
    history_best_p.append(p_values[best_p_i])
    
    progress_bar.set_description(f"π={pi_best:.2f}, ΔQ={avg_delta_q:.1e}")
    
    if episode > min_episodes_run:
        profit_stable = abs(pi_best - prev_best_profit) < eps_pi_threshold
        q_stable = avg_delta_q < stop_delta_q_threshold # 這是最難的一關
        
        if profit_stable and q_stable:
            stable_count += 1
        else:
            stable_count = 0
            
        if stable_count >= stable_episodes_req:
            print(f"\n*** [Fixed] 收斂達成於 Episode {episode} ***")
            print(f"原因: 利潤穩定 且 ΔQ({avg_delta_q:.4f}) < {stop_delta_q_threshold}")
            break
            
    prev_best_profit = pi_best

print(f"最終結果: A={history_best_A[-1]:.3f}, p={history_best_p[-1]:.3f}, π={history_profit[-1]:.4f}")


# ==========================================
# 3. 視覺化與繪圖 (Visualization)
# ==========================================
import matplotlib.pyplot as plt


print("\n繪製 [Fixed] 結果...")
plt.figure(figsize=(15, 10))

# 1. Profit
plt.subplot(2, 2, 1)
plt.plot(history_profit, color='blue')
plt.title(f'[Fixed] Profit (Final: {history_profit[-1]:.4f})')
plt.grid(True)

# 2. Delta Q (Log Scale)
plt.subplot(2, 2, 2)
plt.plot(history_delta_q, color='orange', linewidth=0.5)
plt.yscale('log')
plt.title('[Fixed] Delta Q (Log Scale)')
plt.axhline(y=stop_delta_q_threshold, color='r', linestyle='--', label='Stop Threshold')
plt.legend()
plt.grid(True)

# 3. A Trajectory
plt.subplot(2, 2, 3)
plt.plot(history_best_A, color='green')
plt.title(f'[Fixed] Best A (Final: {history_best_A[-1]:.2f})')
plt.grid(True)

# 4. p Trajectory
plt.subplot(2, 2, 4)
plt.plot(history_best_p, color='red')
plt.title(f'[Fixed] Best p (Final: {history_best_p[-1]:.2f})')
plt.grid(True)

plt.tight_layout()
plt.show()
