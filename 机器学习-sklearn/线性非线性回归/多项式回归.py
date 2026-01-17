import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures as PF

rnd = np.random.RandomState(42) #设置随机数种子
X = rnd.uniform(-3, 3, size=100) #random.uniform，从输入的任意两个整数中取出size个随机数
#生成y的思路：先使用NumPy中的函数生成一个sin函数图像，然后再人为添加噪音
y = np.sin(X) + rnd.normal(size=len(X)) / 3 #random.normal，生成size个服从正态分布的随机数
#使用散点图观察建立的数据集是什么样子
plt.scatter(X, y,marker='o',c='k',s=20)
plt.show()
X=X.reshape(-1, 1) #将X变成n行1列的二维数组
#使用原始数据进行建模
LinearR = LinearRegression().fit(X, y)
TreeR = DecisionTreeRegressor(random_state=0).fit(X, y)
#放置画布
fig, ax1 = plt.subplots(1)
#创建测试数据：一系列分布在横坐标上的点
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
#将测试数据带入predict接口，获得模型的拟合效果并进行绘制
ax1.plot(line, LinearR.predict(line), linewidth=2, color='green',
label="linear regression")
ax1.plot(line, TreeR.predict(line), linewidth=2, color='red',
label="decision tree")
#将原数据上的拟合绘制在图像上
ax1.plot(X[:, 0], y, 'o', c='k')
#其他图形选项
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Result before discretization")
plt.tight_layout()
plt.show()
'''可以看出，线性回归模型无法很好地拟合数据，而决策树模型则表现得非常好。'''

line=np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1) #创建测试数据：一系列分布在横坐标上的点
LinearR = LinearRegression().fit(X, y)
print(LinearR.score(X, y))
print(LinearR.score(line,np.sin(line)))
d=5

poly =PF(d) #实例化多项式特征生成器
X_=poly.fit_transform(X) #对原始数据进行多项式特征转换
line_=poly.transform(line) #对测试数据进行多项式特征转换

LinearR_ = LinearRegression().fit(X_, y) #使用多项式特征进行线性回归建模
print(LinearR_.score(X_, y))
print(LinearR_.score(line_,np.sin(line)))

def polynomial_expression(reg, poly, feature_names=None, tol=1e-12, ndigits=6):
    """
    Convert a trained LinearRegression `reg` that was fit on polynomial features
    produced by `poly` (PolynomialFeatures) into a human-readable polynomial expression.

    - reg: trained LinearRegression
    - poly: fitted PolynomialFeatures instance
    - feature_names: list of original input feature names (optional)
    """
    # determine input feature names
    if feature_names is None:
        # sklearn versions expose the input feature count under different names
        n_in = None
        for attr in ("n_input_features_", "n_features_in_"):
            if hasattr(poly, attr):
                n_in = getattr(poly, attr)
                break
        if n_in is None:
            # poly.powers_ shape is (n_output_features, n_input_features)
            n_in = poly.powers_.shape[1]
        feature_names = [f"x{i}" for i in range(n_in)]

    # feature names after polynomial transform (includes '1' if include_bias=True)
    feat_names = list(poly.get_feature_names_out(feature_names))

    coefs = np.ravel(reg.coef_)
    intercept = float(np.ravel(reg.intercept_)[0]) if np.size(reg.intercept_) > 1 else float(reg.intercept_)

    # If PolynomialFeatures added a bias column named '1', fold its coefficient into intercept
    if '1' in feat_names:
        idx = feat_names.index('1')
        intercept += float(coefs[idx])
        # remove the bias term from lists
        feat_names.pop(idx)
        coefs = np.delete(coefs, idx)

    terms = []
    if abs(intercept) > tol:
        terms.append(f"{intercept:.{ndigits}f}")

    for c, name in zip(coefs, feat_names):
        if abs(c) <= tol:
            continue
        sign = '+' if c > 0 else '-'
        terms.append(f"{sign} {abs(c):.{ndigits}f}*{name}")

    if not terms:
        return '0'

    expr = ' '.join(terms)
    expr = expr.lstrip('+ ').replace('+ -', '- ')
    return expr


# 打印可读的多项式表达式（poly 转换时的原始特征名自动生成 x0, x1...）
print("\n=== Polynomial regression expression (approx) ===")
try:
    expr = polynomial_expression(LinearR_, poly)
    print(expr)
except Exception as e:
    print("Failed to create expression:", e)

# 打印每个多项式特征对应的名称与系数，便于调试
n_in = None
for attr in ("n_input_features_", "n_features_in_"):
    if hasattr(poly, attr):
        n_in = getattr(poly, attr)
        break
if n_in is None:
    n_in = poly.powers_.shape[1]
input_names = [f"x{i}" for i in range(n_in)]
feat_names = poly.get_feature_names_out(input_names)
print('\nFeature names and coefficients:')
for name, coef in zip(feat_names, np.ravel(LinearR_.coef_)):
    print(f"{name}: {coef:.6f}")


fig, ax1 = plt.subplots(1)
ax1.plot(line, LinearR.predict(line), linewidth=2, color='green'
         ,label="linear regression")
ax1.plot(line, LinearR_.predict(line_), linewidth=2, color='red'
         ,label="Polynomial regression")
ax1.plot(X[:, 0], y, 'o', c='k')
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Linear Regression ordinary vs poly")
plt.tight_layout()
plt.show()