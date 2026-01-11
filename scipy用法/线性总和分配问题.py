from scipy.optimize import linear_sum_assignment
import numpy as np
'''
线性总和分配问题是最著名的组合优化问题之一。给定一个“成本矩阵” 
，问题是选择
1.
每行中恰好一个元素
2.
不从任何列中选择超过一个元素
3.
以使所选元素的总和最小化
'''
cost = np.array([[43.5, 45.5, 43.4, 46.5, 46.3],
                 [47.1, 42.1, 39.1, 44.1, 47.8],
                 [48.4, 49.6, 42.1, 44.5, 50.4],
                 [38.2, 36.8, 43.2, 41.2, 37.2]])

row_idx,col_idx=linear_sum_assignment(cost)

styles_arr = np.array(["backstroke", "breaststroke", "butterfly", "freestyle"])
students_arr = np.array(["A", "B", "C", "D", "E"])

styles=styles_arr[row_idx]
students=students_arr[col_idx]

ans=dict(zip(styles,students))
print(ans)

print(f"最小总花费: {cost[row_idx, col_idx].sum()}")