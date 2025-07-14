import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

file_path = r'F:\projects\one\huatu\retrieval_num\aaai\26hate\data\hard.csv'
# style.use('seaborn-v0_8-whitegrid')

try:
    # 使用 pandas 读取 CSV 文件
    data = pd.read_csv(file_path)
    # # 从读取的数据中提取每一列
    # H = data['H'].tolist()
    # Dice = data['Dice'].tolist()
    # Jaccard = data['Jaccard'].tolist()
    # HD = data['HD'].tolist()
    # ASD = data['ASD'].tolist()
    # print(f"成功从 {file_path} 加载数据。")
except FileNotFoundError:
    print(f"错误: 找不到文件 '{file_path}'。请确保文件名正确。")
    # 使用原始的硬编码数据作为备用，以便代码仍然可以运行
    print("将使用代码中的备用数据进行绘图。")
    H = [0.2, 0.4, 0.6, 0.8, 1.0]
    Dice = [87.84, 88.14, 87.82, 88.02, 87.40]
    Jaccard = [79.02, 79.48, 79.00, 79.34, 78.43]
    HD = [4.12, 3.42, 4.73, 3.40, 3.69]
    ASD = [1.25, 0.97, 0.98, 1.03, 1.12]
except KeyError as e:
    print(f"错误: CSV 文件中缺少必需的列: {e}。")
    print("请确保您的 CSV 文件包含 'H', 'Dice', 'Jaccard', 'HD', 'ASD' 这些列。")
    exit()


# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 创建子图
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1.4, 0.6]}, figsize=(4.8, 4))
fig.subplots_adjust(hspace=0.3)

# 1. 获取原始的X轴和Y轴数据
x_original = data['Step']
y1_original = data['hard']
y2_original = data['ours']

# 2. 创建更密集的X轴点，用于绘制平滑曲线
# 我们生成300个点，这样曲线会非常平滑
x_smooth = np.linspace(x_original.min(), x_original.max(), 300)

# 3. 创建样条插值函数
# k=3 表示创建一个三次样条，这对于平滑曲线效果很好
spline1 = make_interp_spline(x_original, y1_original, k=3)
spline2 = make_interp_spline(x_original, y2_original, k=3)

# 4. 计算出平滑后的Y轴数据
y1_smooth = spline1(x_smooth)
y2_smooth = spline2(x_smooth)


# 绘制各个子图的数据
line1, = ax1.plot(x_smooth, y1_smooth, label='CAN', markersize=7, markerfacecolor='#e75041', markeredgecolor='black',
                  color='#e75041', linewidth=1.5)
line2, = ax2.plot(x_smooth, y2_smooth, label='SCAN', markersize=7, markerfacecolor='#5087f6',
                  markeredgecolor='black', color='#5087f6', linewidth=1.5)
# line3, = ax3.plot(x_pos, HD, label='95HD', marker='o', markersize=12, markerfacecolor='#f67295',
#                   markeredgecolor='black', color='black', linewidth=1.5)
# line4, = ax4.plot(x_pos, ASD, label='ASD', marker='^', markersize=12, markerfacecolor='#98dba8',
#                   markeredgecolor='black', color='black', linewidth=1.5)

# 设置刻度和字体大小
for ax in [ax1, ax2]:
    ax.tick_params(axis='y', labelsize=19)
    ax.tick_params(axis='x', labelsize=19)
    ax.grid(True, which='major', linestyle="--", alpha=0.8)

# 设置Y轴范围 (您可以根据需要调整或注释掉以使用自动范围)
ax2.set_ylim(0.2, 1.8)
ax1.set_ylim(15, 60)
ax2.set_yticks(np.arange(0.2, 1.9, 0.8))
ax1.set_yticks(np.arange(15, 61, 10))
# ax3.set_ylim(3.0, 5.2)
# ax4.set_ylim(0.9, 1.5)

# 设置Y轴刻度 (您可以根据需要调整或注释掉以使用自动刻度)
# ax3.set_yticks(np.arange(3.0, 5.3, 0.8))
# ax4.set_yticks(np.arange(1.0, 1.5, 0.2))

# 隐藏不需要的轴标签和边框 (保持您原来的精细设置)
ax1.tick_params(labelbottom=False, bottom=False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax3.spines['bottom'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax4.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
ax1.xaxis.set_ticks_position('none')
ax2.xaxis.set_ticks_position('none')
# ax3.xaxis.set_ticks_position('none')

# # 设置x轴刻度和标签 (从H数据动态生成标签)
# ax4.set_xticks(x_pos)
# ax4.set_xticklabels(H, fontsize=16)

# # 设置图例
# for ax, label in zip([ax1, ax2], ['SCANNER', 'w/o DIV']):
#     ax.legend([label], loc='upper right', bbox_to_anchor=(0.8, 0.7), fontsize=19, frameon=False)
# # for ax, label in zip([ax3, ax4], ['95HD', 'ASD']):
# #     ax.legend([label], loc='lower right', bbox_to_anchor=(1, 0.2), fontsize=19, frameon=False)
ax1.text(0.8, 0.7, r'w/o $\mathcal{L}_{\mathrm{SCAN}}$',
         transform=ax1.transAxes,
         fontsize=19,
        #  color='#ff7300',  # 使用和曲线一致的颜色
         fontweight='bold',
         ha='right',
         va='top')

# 在 ax2 的右上角区域添加文字 'SCANNER'
ax2.text(0.8, 0.7, 'SCANNER',
         transform=ax2.transAxes,
         fontsize=19,
        #  color='#5087f6', # 使用和曲线一致的颜色
         fontweight='bold',
         ha='right',
         va='top')
# 添加破折号断轴标记
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# kwargs.update(transform=ax3.transAxes)
# ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# kwargs.update(transform=ax4.transAxes)
# ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# 设置全局y轴标签
fig.text(0.04, 0.5, 'Gradient Norm', va='center', rotation='vertical', fontsize=19, fontweight='bold')

# 调整布局并保存/显示
plt.tight_layout(rect=[0.05, 0, 1, 1])
fig.subplots_adjust(hspace=0.18)

plt.savefig(r'F:\projects\one\huatu\retrieval_num\aaai\26hate\gradient_en2mm.pdf', bbox_inches='tight', pad_inches=0)
plt.show()