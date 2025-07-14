# cluster_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 19})

csv_path = r"F:\projects\one\huatu\retrieval_num\aaai\26hate\cluster_plots\en2zh\cluster_predictions.csv" 
# csv_path = r"F:\projects\one\huatu\retrieval_num\aaai\26hate\data\cluster_predictions_ours3.csv" 
df = pd.read_csv(csv_path)

# === 可选: 创建输出目录 ===
output_dir = r"F:\projects\one\huatu\retrieval_num\aaai\26hate\cluster_plots\en2zh\nodiv_true"
# output_dir = r"F:\projects\one\huatu\retrieval_num\aaai\26hate\cluster_plots\ours3_"
os.makedirs(output_dir, exist_ok=True)

def smooth_curve(values, window_size=3):
    if len(values) < window_size:
        return values
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode='same')

for cluster_id in sorted(df['cluster'].unique()):
    cluster_df = df[df['cluster'] == cluster_id]

    # # 每 step 计算 prediction_ratio 和 macro-F1
    # def compute_step_metrics(g):
    #     ratio = (g['prediction'] == 1).mean()
    #     try:
    #         f1 = f1_score(g['label'], g['prediction'], average='macro')
    #     except ValueError:
    #         f1 = np.nan  # 如果某类缺失，f1_score 报错
    #     return pd.Series({'prediction_ratio': ratio, 'macro_f1': f1})

    # metrics_df = cluster_df.groupby('step').apply(compute_step_metrics).reset_index()
    # 按 step 计算 prediction_ratio 和 accuracy
    metrics_df = (
        cluster_df.groupby('step')
        .apply(lambda g: pd.Series({
            'prediction_ratio': (g['prediction'] == 1).mean(),
            'macro_f1': f1_score(g['label'], g['prediction'], average='macro')
        }))
        .reset_index()
    )

    x = metrics_df['step'].values
    y_ratio_raw = metrics_df['prediction_ratio'].values
    y_f1_raw = metrics_df['macro_f1'].values

    # y_ratio_smooth = smooth_curve(y_ratio_raw, window_size=2)
    y_ratio_smooth = y_ratio_raw
    y_f1_smooth = smooth_curve(y_f1_raw, window_size=2)

    # === 绘图（共用左轴）===
    plt.figure(figsize=(4.8, 4))
    plt.plot(x, y_ratio_smooth, marker='o', linestyle='-', linewidth=1.5, color='#9999ff', label='Prediction Ratio')
    plt.plot(x, y_f1_smooth, marker='x', linestyle='--', linewidth=1.5, color='#008000', label='Macro-F1')
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Ratio = 0.5')

    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.title(f"Cluster {cluster_id} - Smoothed Prediction Ratio & Macro-F1")
    # plt.xlabel("Online Batch", fontweight='bold')
    # plt.ylabel("Score", fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.yticks(np.arange(0, 1.01, 0.2))
    plt.xticks(np.arange(min(x), max(x)+1, 2))
    plt.grid(True, which='major', linestyle="--", alpha=0.8)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.08), frameon=False)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"cluster_{cluster_id}_ratio_macro_f1_plot_en2zh.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved: {save_path}")