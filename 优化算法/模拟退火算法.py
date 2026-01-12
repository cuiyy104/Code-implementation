import numpy as np
import matplotlib.pyplot as plt
from sko.SA import SA

np.random.seed(42)
n_items=20
weight = np.random.randint(5,15,size=n_items)  # 物品重量
volumes = np.random.randint(3,10,size=n_items)  # 物品体积
value = np.random.randint(10,100,size=n_items)   # 物品价值

max_weight=80
max_volume=60

def func(x):
    total_weight = np.sum(x * weight)
    total_volume = np.sum(x * volumes)
    total_value = np.sum(x * value)

    # 约束：超重或超体积则惩罚（价值置0）
    if total_weight > max_weight or total_volume > max_volume:
        return 0  # SA最小化，返回0相当于惩罚（原本要最大化价值）

    return -total_value  # 取负，让SA最小化等价于原问题最大化

def advance_neighbor(x):
    """生成邻域解的函数，通过随机翻转一个物品的选择状态"""
    x_new = x.copy()
    n=len(x_new)
    op_type=np.random.choice(['single','double','reset'],p=[0.7,0.2,0.1])

    if op_type == 'single':
        idx=np.random.randint(0,n)
        x_new[idx]=1 - x_new[idx]  # 翻转选择状态
    elif op_type == 'double':
        idx1,idx2=np.random.choice(n,size=2,replace=False)
        x_new[idx1]=1 - x_new[idx1]
        x_new[idx2]=1 - x_new[idx2]
    else:  # reset
        reset_num=max(1,int(n*0.2))
        reset_idx=np.random.choice(n,size=reset_num,replace=False)
        x_new[reset_idx]=1 - x_new[reset_idx]
    return x_new

x0=np.random.randint(0,2,size=n_items)
sa=SA(func=func,
      x0=x0,
      T_max=500,
      T_min=1e-6,
      L=500,
      max_stay_counter=500,
      neighbor=advance_neighbor
     )
best_x,best_y=sa.run()

best_x = (best_x>0.5).astype(int)
# 计算最终结果
total_selected = np.sum(best_x)
total_weight = np.sum(best_x * weight)
total_volume = np.sum(best_x * volumes)
total_value = -best_y  # 还原为原问题的最大化价值（扣除惩罚后）

# 输出详细结果
print("="*50)
print("复杂0-1背包问题 SA 求解结果")
print("="*50)
print(f"选中的物品索引（从0开始）：{np.where(best_x==1)[0]}")
print(f"选中物品数量：{total_selected}")
print(f"总重量：{total_weight} (最大承重：{max_weight})")
print(f"总容积：{total_volume} (最大容积：{max_volume})")
print(f"最大总价值：{total_value}")
print(f"是否满足所有约束：{'是' if (total_weight<=max_weight and total_volume<=max_volume) else '否'}")

plt.figure(figsize=(10, 6))
# 绘制最优价值收敛曲线（还原为正值）
plt.plot(-np.array(sa.best_y_history), color='#2E86AB', linewidth=2)
plt.xlabel("epoch", fontsize=12)
plt.ylabel("best_val", fontsize=12)
plt.title("process", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()