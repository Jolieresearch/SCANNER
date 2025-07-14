import matplotlib.pyplot as plt
import numpy as np

mmd_results = {
    "cluster1": [
        0.02252357,  # with Gaussian
        0.151416525,  # with FakeSV (symmetric)
    ],
    "cluster2": [
        0.0329,  # with Gaussian
        0.337943,  # with FakeTT (symmetric)
    ],
    "cluster3": [
        0.038620,  # with Gaussian
        0.38103,  # with FakeSV (symmetric)
    ],
}

# Global font size settings
fontsize = 16
plt.rcParams.update({"font.size": fontsize})
plt.rcParams.update({"font.family": "Times New Roman"})

# Create figure with specified size
fig, ax = plt.subplots(figsize=(4, 2.4))

# Extract data for plotting
methods = list(mmd_results.keys())
values = [mmd_results[method] for method in methods]

# Set up bar positions
x = np.arange(len(methods))  # 控制 cluster1, 2, 3 之间更近

width = 0.28  # Width of bars (increased slightly)
spacing = 0.08  # Gap between bars within each group

# Better color scheme - using more visually appealing colors
# colors = ["#b2df8a", "#a6cee3", "#1f78b4"]  # Blue, Red, Green
colors = ["#ece2f0", "#a6bddb"]

# Create bars for each value within each method
bars1 = ax.bar(
    x - width - spacing,
    [val[1] for val in values],
    width,
    label="Target",
    color=colors[0],
    edgecolor="black",
    linewidth=1,
)
bars2 = ax.bar(
    x,
    [val[0] for val in values],
    width,
    label="Centroid",
    color=colors[1],
    edgecolor="black",
    linewidth=1,
)


# Set labels and formatting
# ax.set_xlabel("Methods", labelpad=0)
ax.set_ylabel("Entropy", labelpad=0)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=fontsize - 2)
ax.tick_params(axis="y", labelsize=fontsize - 2)

y_min, y_max = ax.get_ylim()
ax.set_ylim(0, 0.39)
ax.set_yticks(np.arange(0, 0.38, 0.6))

# # Add legend with specified parameters
# ax.legend(
#     loc="upper left",
#     handleheight=0.5,
#     labelspacing=0.1,
#     columnspacing=0.5,
#     borderpad=0.1,
#     handletextpad=0.1,
#     borderaxespad=0.1,
#     frameon=False,
#     # ncol=2
# )
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=[handles[1], handles[0]],  # ← 交换顺序
    labels=[labels[1], labels[0]],
    loc="upper left",
    handleheight=0.6,
    labelspacing=0.1,
    columnspacing=0.5,
    borderpad=0.1,
    handletextpad=0.1,
    borderaxespad=0.1,
    frameon=False
)

# Use tight layout and save
plt.tight_layout()
plt.savefig(r"F:\projects\one\huatu\retrieval_num\aaai\26hate\entropy\entropy_en2mm.pdf", bbox_inches="tight", pad_inches=0.0)
print("Saved figure to statis/fig/exp-pre-bar.pdf")
plt.show()