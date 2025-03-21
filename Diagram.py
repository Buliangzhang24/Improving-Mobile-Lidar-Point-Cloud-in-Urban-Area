import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['Initial MLS','KNN on Voxel',
           'Random Sampling with Bayesian Regression', 'KNN on Patch',
           'Normal-Based Guided Filtering', 'Normal-Based Bilateral Filtering',
           'Normal Voting Tensor', 'Random Sampling with Kernel Density']
roof_scores = [0.316, 0.288, 0.316, 0.316, 0.316, 0.316, 0.316, 0.363]  # 右侧Roof Score
road_scores = [0.770, 0.770, 0.771, 0.677, 0.771, 0.771, 0.771, 0.771]  # 左侧Road Score

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 设置y轴位置
y = np.arange(len(methods))

# 左侧柱状图（Road Score，向左延伸）
left_colors = ['blue'] * len(methods)
left_colors[methods.index('Initial MLS')] = 'green'  # 将Initial MLS的颜色改为绿色
left_bars = ax.barh(y, -np.array(road_scores), height=0.4, color=left_colors, label='Road Score')

# 右侧柱状图（Roof Score，向右延伸）
right_colors = ['red'] * len(methods)
right_colors[methods.index('Initial MLS')] = 'green'  # 将Initial MLS的颜色改为绿色
right_bars = ax.barh(y, roof_scores, height=0.4, color=right_colors, label='Roof Score', alpha=0.5)

# 设置y轴标签（方法名称）
ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.set_xlabel('Score')
ax.set_title('Comparison of Road and Roof Scores')

# 将y轴标签（方法名称）放在中间
ax.yaxis.set_label_coords(-0.1, 0.5)

# 设置x轴范围，使左右对称
ax.set_xlim(-1.0, 1.0)

# 隐藏x轴的负值标签
ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels([0.8, 0.6, 0.4, 0.2, 0, 0.2, 0.4, 0.6, 0.8])

# 在左侧柱状图上显示数值
for bar, score in zip(left_bars, road_scores):
    ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2, f'{score:.3f}',
            va='center', ha='right', color='black', fontsize=9)

# 在右侧柱状图上显示数值
for bar, score in zip(right_bars, roof_scores):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f'{score:.3f}',
            va='center', ha='left', color='black', fontsize=9)

# 显示图例
ax.legend()

plt.savefig('D:/E_2024_Thesis/Paper/Final Paper/LaTex/Score_Com.png', dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.show()