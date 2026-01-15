from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

'''
当真实标签已知时
有三类评估指标：
1. 互信息分数（Mutual Information Score）
包括：
   - 归一化互信息（Normalized Mutual Information, NMI）
   metrics.normalized_mutual_info_score (y_pred, y_true)
   
   - 调整互信息（Adjusted Mutual Information, AMI）
   metrics.mutual_info_score (y_pred, y_true) 
   以上均是越接近1越好
2. 调整兰德指数（Adjusted Rand Index）
metrics.adjusted_rand_score (y_pred, y_true)
越接近1越好
3.V-measure分数：
包括：
    - 同质性（Homogeneity）
    - 完整性（Completeness）
    - V-measure
    越接近1越好
'''

'''接下来是标签未知的情况（常用于无监督聚类评估）'''
# 轮廓系数（Silhouette Coefficient）示例
# 我们用 make_blobs 生成一个用于演示的二维数据集 X，4 个簇，500 个样本
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

# 使用 KMeans 将数据聚成 3 类（注意：这里故意用 3 个簇来演示评估指标的差异）
cluster = KMeans(n_clusters=3, random_state=0).fit(X)
# cluster.labels_ 是每个样本的簇标签（长度 = n_samples）
y_pred = cluster.labels_

# silhouette_score 返回所有样本轮廓系数的平均值（标量），范围约为 [-1, 1]
# 较接近 1 表示簇分离良好且内部紧凑；接近 -1 表示样本可能被错误聚到当前簇
print(f'轮廓系数: {silhouette_score(X, y_pred)}')

# 卡林斯基-哈拉巴斯指数（Calinski-Harabasz Index）
# 该指标越高越好，衡量簇的密集度和簇间离散度之比
from sklearn.metrics import calinski_harabasz_score
print(f'卡林斯基指数: {calinski_harabasz_score(X, y_pred)}')


'''基于轮廓系数选择合适的 n_clusters（循环计算并可视化每个样本的轮廓系数分布）'''
# 这里我们尝试多种聚类数量 2..7，绘制每种情况下的轮廓分析图
for n_clusters in [2, 3, 4, 5, 6, 7]:
    # n_clusters 代表当前要评估的簇数量
    n_clusters = n_clusters

    # 创建一个包含两个子图的图像，ax1 用于绘制 silhouette plot（每个样本的轮廓系数分布）
    # ax2 用于绘制聚类结果在二维特征空间（X 的前两列）上的散点图
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # 设置整张图的大小（英寸），便于在屏幕上查看
    fig.set_size_inches(18, 7)

    # ax1 的横轴表示轮廓系数取值范围（理论上在 -1 到 1），这里设为 [-0.1, 1] 便于显示
    ax1.set_xlim([-0.1, 1])
    # 纵轴用于把不同簇的样本分段放置，每个簇之间留出固定间隔（10）���增强可读性
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])

    # 对当前数据 X 执行 KMeans 聚类，得到簇分配
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    # cluster_labels 是一个一维数组，保存每个样本所属的簇索引（0..n_clusters-1）
    cluster_labels = clusterer.labels_

    # 计算当前聚类结果的平均轮廓系数（标量），便于比较不同 n_clusters 的整体质量
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # 计算每个样本的轮廓系数（数组，长度 = n_samples），用于绘制左侧的密度条形图
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # y_lower 用于在纵轴定位当前簇的开始位置（初始留 10 个单位的空白）
    y_lower = 10

    # 遍历每个簇并绘制该簇中所有样本的轮廓系数（按值升序排序后绘制，视觉上呈现为条带）
    for i in range(n_clusters):
        # 取出属于第 i 个簇的所有样本的轮廓系数
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        # 对这些轮廓系数排序，这样绘出的条带会从小到大排列，更容易观察簇内分布
        ith_cluster_silhouette_values.sort()

        # size_cluster_i 是第 i 个簇中样本的个数（整数）
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        # y_upper 表示当前簇条带的结束位置（不含），用于在纵轴上分段堆叠各簇
        y_upper = y_lower + size_cluster_i

        # 根据簇索引 i 生成颜色（nipy_spectral 色图按比例生成颜色），以便不同簇颜色不同
        color = cm.nipy_spectral(float(i) / n_clusters)

        # 在 ax1 上绘制当前簇的轮廓系数条带：纵向范围是 y_lower..y_upper，横向长度按轮廓系数值
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          , ith_cluster_silhouette_values
                          , facecolor=color
                          , alpha=0.7
                          )

        # 在条带左侧标注该簇的编号（文本位置垂直居中）
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , str(i))

        # 更新 y_lower，为下一个簇的条带留出 10 单位的间隔
        y_lower = y_upper + 10

    # 左图的标题与坐标标签
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # 在左图绘制一条垂直虚线，显示全局平均轮廓系数，方便对比各簇的条带相对位置
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    # 不显示 y 轴刻度（因为纵轴只是用来分隔簇，刻度没有实际含义）
    ax1.set_yticks([])
    # 指定 x 轴的刻度以便更直观地读取轮廓系数数值
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 右图：将所有样本按簇着色后绘制散点图（仅用 X 的前两列作为可视化坐标）
    # 注意：如果 X 的维度 > 2，这里只是用前两维做投影展示，不代表高维空间的真实分布
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1]
                , marker='o'
                , s=8
                , c=colors
                )

    # 获取簇中心并在散点图上以 'x' 标出（红色，较大尺寸）
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
                c="red", alpha=1, s=200)

    # 右图的标题与坐标标签（说明这是特征空间的前两维）
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    # 整体标题：在上方标明当前绘制的是哪个 n_clusters 的轮廓分析
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    # 显示图像
    plt.show()