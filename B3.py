import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# (B.3 New) Stackelberg: Shared Ad Fee
# ==========================================

# ------------------- 參數設定 -------------------
n_episodes = 100000   
gamma = 0.9          

alpha_r = 1.0        
alpha_s = 0.05       

epsilon_start = 0.5
epsilon_end = 0.01   
decay_rate = 0.9999  

# --- 收斂判斷參數 ---
min_episodes = 5000         
convergence_window = 1000   
stability_tolerance = 0.005 
stable_counter = 0          
stop_threshold = 20         
history_pi_s = []           
# --------------------

beta = 10.0
epsA = 0.5
epsp = -1.5
c = 2

max_w = 15; min_w = c + 1
max_p = 30; min_p = c + 1
# 設定各別廣告上限
max_A_individual = 10 
min_A_individual = 1  

penalty = -10.0 

# ------------------- 動作空間定義 -------------------
# Supplier 決定 (w, As)
possible_supplier_actions = []
for w in range(min_w, max_w + 1):
    for As in range(min_A_individual, max_A_individual + 1):
        possible_supplier_actions.append((w, As))

# Retailer 決定 (p, Ar)
possible_retailer_actions = []
for p in range(min_w, max_p + 1):
    for Ar in range(min_A_individual, max_A_individual + 1):
        possible_retailer_actions.append((p, Ar))

n_actions_s = len(possible_supplier_actions)
n_actions_r = len(possible_retailer_actions)

# Q-Tables
Q_supplier = np.zeros(n_actions_s)
# Retailer 的 Q 表依賴於 Supplier 的 (w, As)
Q_retailer = np.zeros((n_actions_s, n_actions_r))

# ------------------- 函數 -------------------
def get_profits(w, As, p, Ar):
    if p <= w: return penalty, penalty
    
    A_total = As + Ar
    if A_total <= 0: return penalty, penalty # 總廣告量不能為 0
    
    Q = beta * (A_total ** epsA) * (p ** epsp)
    
    # B3: 共同分擔
    pi_s = (w - c) * Q - As
    pi_r = (p - w) * Q - Ar
    
    return pi_s, pi_r

# ------------------- 主訓練 -------------------
best_results = {'w': 0, 'p': 0, 'As': 0, 'Ar': 0, 'pi_s': -np.inf, 'pi_r': -np.inf}
epsilon = epsilon_start

print("【B3 New: Stackelberg (Shared Ad)】Training Start...")
pbar = tqdm(range(n_episodes))

for episode in pbar:
    epsilon = max(epsilon_end, epsilon * decay_rate)

    # 1. Supplier Move (選 w, As)
    if random.random() < epsilon:
        idx_s = random.randint(0, n_actions_s - 1)
    else:
        idx_s = np.argmax(Q_supplier)
    w, As = possible_supplier_actions[idx_s]
    
    # 2. Retailer Thinking Phase (針對目前的 w, As 找出最佳 p, Ar)
    n_thinking_steps = 1000
    
    # 記憶保護
    current_best_r_idx = np.argmax(Q_retailer[idx_s])
    p_best, Ar_best = possible_retailer_actions[current_best_r_idx]
    _, pi_best_check = get_profits(w, As, p_best, Ar_best)
    if pi_best_check > 0:
        Q_retailer[idx_s, current_best_r_idx] = pi_best_check

    for _ in range(n_thinking_steps):
        rand_r_idx = random.randint(0, n_actions_r - 1)
        p_try, Ar_try = possible_retailer_actions[rand_r_idx]
        
        _, pi_r_try = get_profits(w, As, p_try, Ar_try)
        
        if pi_r_try > 0:
            Q_retailer[idx_s, rand_r_idx] = pi_r_try
        else:
            Q_retailer[idx_s, rand_r_idx] = penalty

    # 3. Retailer Execution
    idx_r = np.argmax(Q_retailer[idx_s])
    p, Ar = possible_retailer_actions[idx_r]
    
    # 4. Interaction
    pi_s, pi_r = get_profits(w, As, p, Ar)
    
    # 生存檢查
    if pi_r <= 0.05:
        r_s = penalty
    else:
        r_s = pi_s
        if pi_s > best_results['pi_s']:
            best_results = {'w': w, 'p': p, 'As': As, 'Ar': Ar, 'pi_s': pi_s, 'pi_r': pi_r}

    # Supplier Update
    Q_supplier[idx_s] += alpha_s * (r_s - Q_supplier[idx_s])

    # 5. 收斂檢查
    history_pi_s.append(r_s)
    if episode > min_episodes:
        recent_avg = np.mean(history_pi_s[-convergence_window:])
        prev_avg = np.mean(history_pi_s[-convergence_window*2 : -convergence_window])
        
        if abs(recent_avg - prev_avg) < stability_tolerance and recent_avg > 0 and epsilon < 0.1:
            stable_counter += 1
        else:
            stable_counter = 0
            
        if stable_counter > stop_threshold:
            print(f"\n✅ Converged at episode {episode}!")
            print(f"   Stable Average Profit: {recent_avg:.4f}")
            break

    if episode % 1000 == 0:
        pbar.set_description(f"Eps: {epsilon:.2f} | Best S_Profit: {best_results['pi_s']:.2f}")

# ------------------- 結果 -------------------
print("-" * 30)
print("【B3 New 最終結果】")
print(f"總訓練回數: {episode + 1}")
print(f"批發價 w: {best_results['w']}")
print(f"售價 p  : {best_results['p']}")
print(f"廣告 As : {best_results['As']} (Supplier)")
print(f"廣告 Ar : {best_results['Ar']} (Retailer)")
print(f"供應商利潤: {best_results['pi_s']:.4f}")
print(f"零售商利潤: {best_results['pi_r']:.4f}")
print("-" * 30)