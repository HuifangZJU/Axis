import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_noise_heatmap_from_summary(
    data,
    metric_label="Top-k label consistency",
    method_order=None,
    save_path=None,
    figsize=(7.2, 4.8),
    cmap="viridis",
    annotate=True,
    highlight_method="Ours",
):
    """
    data format:
        {
            "Unaligned": [0.4406, 0.4406, 0.4406, 0.4406, 0.4406, 0.4406],
            ...
            "Ours":      [0.5025, 0.4843, 0.4826, 0.4804, 0.4873, 0.4729],
        }
    """

    # build dataframe
    df = pd.DataFrame(data).T
    df.columns = [f"L{i}" for i in range(df.shape[1])]
    df.index.name = "Method"

    if method_order is not None:
        method_order = [m for m in method_order if m in df.index]
        df = df.reindex(method_order)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=".3f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": metric_label},
    )

    ax.set_xlabel("Noise level")
    ax.set_ylabel("")
    ax.set_title("Noise robustness heatmap", pad=10)

    # nicer x tick labels
    ax.set_xticklabels([str(i) for i in range(df.shape[1])], rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # highlight AXIS / Ours row
    if highlight_method in df.index:
        row_idx = list(df.index).index(highlight_method)
        rect = Rectangle(
            (0, row_idx),
            width=df.shape[1],
            height=1,
            fill=False,
            edgecolor="crimson",
            linewidth=2.5,
            clip_on=False,
        )
        ax.add_patch(rect)

    # color the highlighted method label
    for label in ax.get_yticklabels():
        if label.get_text() == highlight_method:
            label.set_color("crimson")
            label.set_fontweight("bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()




# -------------------------
# Input data
# -------------------------
data = {
    "Unaligned": [0.4406, 0.4406, 0.4406, 0.4406, 0.4406, 0.4406],
    "SimpleITK": [0.4835, 0.4878, 0.4818, 0.3856, 0.3394, 0.3806],
    "VoxelMorph": [0.4737, 0.4518, 0.4578, 0.4340, 0.4296, 0.3965],
    "NiceTrans": [0.5003, 0.4860, 0.4757, 0.4392, 0.3927, 0.3656],
    "PASTE": [0.4638, 0.4638, 0.4638, 0.4638, 0.4638, 0.4638],
    "GPSA": [0.4641, 0.4641, 0.4641, 0.4641, 0.4641, 0.4641],
    "SANTO": [0.4521, 0.4521, 0.4521, 0.4521, 0.4521, 0.4521],
    "Axis": [0.5025, 0.4843, 0.4826, 0.4804, 0.4873, 0.4729],
}

noise_levels = np.array([0, 1, 2, 3, 4, 5], dtype=int)
metric_name = "Top-k label consistency"

method_order = [
    "Unaligned",
    "PASTE",
    "GPSA",
    "SANTO",
    "SimpleITK",
    "VoxelMorph",
    "NiceTrans",
    "Axis",
]
#
# plot_noise_heatmap_from_summary(
#     data=data,
#     metric_label="Top-k label consistency",
#     method_order=method_order,
#     figsize=(7.2, 4.8),
#     highlight_method="Axis",
# )

# -------------------------
# Convert to dataframe
# -------------------------
df = pd.DataFrame(data, index=noise_levels)
df.index.name = "Noise level"

long_df = (
    df.reset_index()
      .melt(id_vars="Noise level", var_name="Method", value_name="Score")
)

# best competing baseline by mean performance across all noise levels
method_mean = long_df.groupby("Method", as_index=False)["Score"].mean()
best_baseline = (
    method_mean[method_mean["Method"] != "Axis"]
    .sort_values("Score", ascending=False)
    .iloc[0]["Method"]
)

competing_img_baseline="NiceTrans"

# -------------------------
# Style helper
# -------------------------
def method_style(method):
    if method == "Axis":
        return {
            "color": "crimson",
            "lw": 3.2,
            "ms": 8.5,
            "alpha": 1.0,
            "zorder": 6,
        }
    elif method == best_baseline:
        return {
            "color": "royalblue",
            "lw": 2.6,
            "ms": 7.5,
            "alpha": 0.95,
            "zorder": 5,
        }
    elif method ==competing_img_baseline:
        return {
            "color": "orange",
            "lw": 2.6,
            "ms": 7.5,
            "alpha": 0.95,
            "zorder": 5,
        }
    else:
        return {
            "color": "0.65",
            "lw": 1.8,
            "ms": 5.0,
            "alpha": 0.9,
            "zorder": 3,
        }


# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))
plt.rcParams.update({"font.size": 16})
# subtle background bands to indicate stronger corruption
for x0, x1, a in [(0, 1, 0.03), (1, 3, 0.05), (3, 5, 0.08)]:
    ax.axvspan(x0, x1, color="black", alpha=a, lw=0)

for method in df.columns:
    y = df[method].values
    st = method_style(method)

    ax.plot(
        noise_levels,
        y,
        marker="o",
        linewidth=st["lw"],
        markersize=st["ms"],
        color=st["color"],
        alpha=st["alpha"],
        zorder=st["zorder"],
    )

# direct labels at the right
label_order = df.iloc[-1].sort_values(ascending=False).index.tolist()
used_y = []

def nudge_y(y, used, min_gap=0.008):
    y_new = y
    while any(abs(y_new - yy) < min_gap for yy in used):
        y_new -= min_gap
    used.append(y_new)
    return y_new

for method in label_order:
    y_end = df.loc[5, method]
    y_lab = nudge_y(y_end, used_y)

    color = method_style(method)["color"]
    weight = "bold" if method == "Axis" else "normal"

    drop = df.loc[5, method] - df.loc[0, method]
    label_text = f"{method}  ({drop:+.3f})"

    ax.text(
        noise_levels[-1] + 0.15,
        y_lab,
        label_text,
        color=color,
        # fontsize=11,
        fontweight=weight,
        va="center",
    )

# emphasize Ours start and end points
for x in [0, 5]:
    ax.scatter(
        [x],
        [df.loc[x, "Axis"]],
        s=90,
        color="crimson",
        edgecolor="white",
        linewidth=1.0,
        zorder=7,
    )

# axes and layout
ax.set_xlim(-0.1, 6.2)
ax.set_xticks(noise_levels)
ax.set_xlabel("Noise level",fontsize=16)
ax.set_ylabel(metric_name)
ax.set_title("Robustness to input noise", pad=12)

# clean style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.22)
ax.grid(axis="x", alpha=0.08)


plt.tight_layout()
plt.show()