import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

file_path = r'F:\projects\one\huatu\retrieval_num\aaai\26hate\data\hard_en2zh.csv'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 19})

# 读取数据
data = pd.read_csv(file_path)
x = data['Step']
y1 = data['hard']
y2 = data['ours']

# 平滑曲线
x_smooth = np.linspace(x.min(), x.max(), 300)
y1_smooth = make_interp_spline(x, y1, k=3)(x_smooth)
y2_smooth = make_interp_spline(x, y2, k=3)(x_smooth)

# 创建主图
fig, ax = plt.subplots(figsize=(4.8, 4))

# 绘制两条曲线
ax.plot(x_smooth, y1_smooth, color='#e75041', linewidth=1.5)
ax.plot(x_smooth, y2_smooth, color='#5087f6', linewidth=1.5)

# 坐标轴配置
# ax.set_xlabel("Step", fontweight='bold')
# ax.set_ylabel("Gradient Norm", fontweight='bold')
ax.set_ylim(2, 30)
ax.set_yticks(np.arange(2, 31, 5))
ax.set_xticks(np.arange(int(x.min()), int(x.max())+1, 5))

# 样式优化
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', frameon=False)

# 曲线标签（可选）
ax.text(
    0.78, 0.82,
    r'w/o $\mathcal{L}_{\mathrm{SCAN}}$',
    transform=ax.transAxes,
    fontsize=19,
    fontweight='bold',
    ha='right'
)

ax.text(0.78, 0.33, 'SCANNER', transform=ax.transAxes, fontsize=19, fontweight='bold', ha='right')

# 布局 & 保存
plt.tight_layout()
plt.savefig(r'F:\projects\one\huatu\retrieval_num\aaai\26hate\gradient_en2zh.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
