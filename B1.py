import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# (B.1) Stackelberg MARL: Robust & Stable (With Early Stopping)
# ==========================================

# ------------------- 參數設定 -------------------
n_episodes = 100000   
gamma = 0.9          

# 學習率
alpha_r = 1.0        # Retailer: 過目不忘
alpha_s = 0.05       # Supplier: 穩定學習

# Epsilon Decay 參數
epsilon_start = 0.5
epsilon_end = 0.01   
decay_rate = 0.9999  

# ---【新增】收斂判斷參數 ---
min_episodes = 5000         # 至少跑幾輪才開始檢查
convergence_window = 1000   # 取最近 1000 筆的平均
stability_tolerance = 0.005 # 容許誤差 (比舊版嚴格一點，因為這裡利潤數值較小)
stable_counter = 0          # 連續穩定次數計數器
stop_threshold = 20         # 連續 20 次檢查都穩定才停止
history_pi_s = []           # 紀錄歷史利潤
# ---------------------------

beta = 10.0
epsA = 0.5
epsp = -1.5
c = 2

max_w = 15; min_w = c + 1
max_p = 30; min_p = c + 1
max_A = 15; min_A = 1 
penalty = -10.0 

# ------------------- 動作空間定義 -------------------
possible_w = list(range(min_w, max_w + 1)) 

possible_retailer_actions = []
for p in range(min_w, max_p + 1):
    for A in range(min_A, max_A + 1):
        possible_retailer_actions.append((p, A))

n_actions_s = len(possible_w)
n_actions_r = len(possible_retailer_actions)

# Q-Tables
Q_supplier = np.zeros(n_actions_s)
Q_retailer = np.zeros((n_actions_s, n_actions_r))

# ------------------- 函數 -------------------
def get_profits(w, p, A):
    if p <= w: return penalty, penalty 
    Q = beta * (A ** epsA) * (p ** epsp)
    pi_s = (w - c) * Q
    pi_r = (p - w) * Q - A
    return pi_s, pi_r

# ------------------- 主訓練 -------------------
best_results = {'w': 0, 'p': 0, 'A': 0, 'pi_s': -np.inf, 'pi_r': -np.inf}

# 初始化 epsilon
epsilon = epsilon_start

print("【B1: Robust MARL Training with Early Stopping】Start...")
pbar = tqdm(range(n_episodes))

for episode in pbar:
    # 1. 更新 Epsilon
    epsilon = max(epsilon_end, epsilon * decay_rate)

    # --- Supplier Move ---
    if random.random() < epsilon:
        idx_w = random.randint(0, n_actions_s - 1)
    else:
        idx_w = np.argmax(Q_supplier)
    w = possible_w[idx_w]
    
    # --- Retailer Learning Phase (Thinking Loop) ---
    n_thinking_steps = 1000 
    
    # 保持記憶
    current_best_idx = np.argmax(Q_retailer[idx_w])
    p_best, A_best = possible_retailer_actions[current_best_idx]
    _, pi_best_check = get_profits(w, p_best, A_best)
    if pi_best_check > 0:
        Q_retailer[idx_w, current_best_idx] = pi_best_check 

    for _ in range(n_thinking_steps):
        rand_idx = random.randint(0, n_actions_r - 1)
        p_try, A_try = possible_retailer_actions[rand_idx]
        _, pi_r_try = get_profits(w, p_try, A_try)
        
        if pi_r_try > 0: 
            Q_retailer[idx_w, rand_idx] = pi_r_try
        else:
            Q_retailer[idx_w, rand_idx] = penalty

    # --- Retailer Execution ---
    idx_r = np.argmax(Q_retailer[idx_w])
    p, A = possible_retailer_actions[idx_r]
    
    # --- Interaction & Update ---
    pi_s, pi_r = get_profits(w, p, A)
    
    if pi_r <= 0.05: 
        r_s = penalty
    else:
        r_s = pi_s
        if pi_s > best_results['pi_s']:
            best_results = {'w': w, 'p': p, 'A': A, 'pi_s': pi_s, 'pi_r': pi_r}

    # Supplier Update
    Q_supplier[idx_w] += alpha_s * (r_s - Q_supplier[idx_w])
    
    # ---【新增】紀錄歷史與收斂檢查 ---
    history_pi_s.append(r_s)
    
    if episode > min_episodes:
        # 計算最近 N 筆 與 前 N 筆 的平均差異
        recent_avg = np.mean(history_pi_s[-convergence_window:])
        prev_avg = np.mean(history_pi_s[-convergence_window*2 : -convergence_window])
        
        # 條件 1: 變動極小
        # 條件 2: 平均利潤必須是正的 (避免卡在 penalty -10 收斂)
        # 條件 3: Epsilon 必須夠小 (避免還在瞎猜時運氣好連續幾次一樣就停了)
        if abs(recent_avg - prev_avg) < stability_tolerance and recent_avg > 0 and epsilon < 0.1:
            stable_counter += 1
        else:
            stable_counter = 0
            
        if stable_counter > stop_threshold:
            print(f"\n✅ Converged at episode {episode}!")
            print(f"   Stable Average Profit: {recent_avg:.4f}")
            break

    if episode % 1000 == 0:
        pbar.set_description(f"Eps: {epsilon:.2f} | Best w: {best_results['w']}")

# ------------------- 結果 -------------------
print("-" * 30)
print("【B1 MARL 最終結果】")
print(f"總訓練回數: {episode + 1}")
print(f"批發價 w: {best_results['w']}")
print(f"售價 p  : {best_results['p']}")
print(f"廣告 A  : {best_results['A']}")
print(f"供應商利潤: {best_results['pi_s']:.4f}")
print(f"零售商利潤: {best_results['pi_r']:.4f}")
print("-" * 30)

# ------------------- 驗證 (動態 A) -------------------
cand_w = best_results['w']
cand_p = best_results['p']
cand_A = best_results['A']

real_best_pi_r = -np.inf
real_best_p = -1
real_best_A = -1 

for check_p in range(cand_w + 1, max_p + 1):
    for check_A in range(min_A, max_A + 1):
        Q_val = beta * (check_A ** epsA) * (check_p ** epsp)
        val_r = (check_p - cand_w) * Q_val - check_A
        
        if val_r > real_best_pi_r:
            real_best_pi_r = val_r
            real_best_p = check_p
            real_best_A = check_A

print(f"驗證: 在 w={cand_w} 下，數學最佳解 p={real_best_p}, A={real_best_A}, 利潤={real_best_pi_r:.4f}")

if cand_p == real_best_p and cand_A == real_best_A:
    print("✅ 完美收斂 (Perfect Convergence)。")
else:
    print("⚠️ 仍有差距。")