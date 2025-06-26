import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def testSoftmax(visits):
    temp = 0.1
    visits_arr = np.array(visits, dtype=np.float32)
    visits_arr[visits_arr == 0] = 1e-10  # 避免log(0)
    
    log_visits = np.log(visits_arr)
    scaled_logs = 1.0 / temp * log_visits
    return softmax(scaled_logs)

visits = np.arange(0, 21)  # 包含 0 和 20

# 计算概率分布
act_probs = testSoftmax(visits)

# 绘制图表
plt.figure(figsize=(10, 6))

# 使用条形图或散点图
plt.scatter(visits, act_probs, s=100, color='skyblue', alpha=0.7)
plt.plot(visits, act_probs, 'r-', linewidth=1)  # 连线

plt.xlabel('Visits')
plt.ylabel('Probability')
plt.title('Softmax Probability Distribution (Temp=0.001)')
plt.grid(True, linestyle='--', alpha=0.5)

# 添加数值标签
for i, prob in enumerate(act_probs):
    plt.text(visits[i], prob, f'{prob:.2e}', 
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()