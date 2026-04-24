from statistic_plot_utils import *
# ============================================================
# Config
# ============================================================


DISTANCE_ORDER = [10, 300, 310, 320]
DISTANCE_BIN_ORDER = ["Near (10 μm)", "Far (300–320 μm)"]


# ============================================================
# Basic helpers
# ============================================================

def format_distance_label(x):
    """
    10 -> 10um
    300 -> 300um
    """
    try:
        return f"{int(float(x))}um"
    except Exception:
        return str(x)


def format_domain_label(x):
    """
    1.0 -> L3
    2.0 -> L4
    3.0 -> L5
    4.0 -> L6
    5.0 -> WM
    """
    mapping = {
        "1": "L3",
        "1.0": "L3",
        1: "L3",
        1.0: "L3",
        "2": "L4",
        "2.0": "L4",
        2: "L4",
        2.0: "L4",
        "3": "L5",
        "3.0": "L5",
        3: "L5",
        3.0: "L5",
        "4": "L6",
        "4.0": "L6",
        4: "L6",
        4.0: "L6",
        "5": "WM",
        "5.0": "WM",
        5: "WM",
        5.0: "WM",
    }
    return mapping.get(x, mapping.get(str(x), str(x)))


def load_metrics_csv(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)


    # missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    # if missing:
    #     raise ValueError(
    #         f"CSV is missing required columns: {missing}\n"
    #         f"Available columns: {list(df.columns)}"
    #     )

    df = df.copy()
    df["method"] = df["method"].astype(str).str.lower()
    df["domain"] = df["domain"].astype(str)

    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["pair_id"] = (
        "g" + df["group_id"].astype(str)
        + ":" + df["fixed_id"].astype(str)
        + "→" + df["moving_id"].astype(str)
    )
    df["pair_type"] = (
        df["fixed_id"].astype(str) + "→" + df["moving_id"].astype(str)
    )

    # df["distance_bin"] = np.where(
    #     df["distance_um"] == 10,
    #     "Near (10 μm)",
    #     "Far (300–320 μm)"
    # )
    # print(df.columns)
    # test = input()

    return df


# ============================================================
# Example usage: run one figure at a time
# ============================================================

if __name__ == "__main__":
    # CSV_PATH = "/media/huifang/data/registration/DLPFC/evaluation/dlpfc_domain_metrics_all_methods.csv"
    # CSV_PATH = "/media/huifang/data/registration/xenium/mouse_brain/evaluation/mouse_time_domain_metrics_all_methods.csv"
    # CSV_PATH = "/media/huifang/data/registration/breast_cancer/evaluation/breast_cancer_domain_metrics_all_methods.csv"
    CSV_PATH = "/media/huifang/data/registration/SCC/evaluation/scc_molecular_metrics_all_methods.csv"

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
    )

    df = load_metrics_csv(CSV_PATH)

    # --------------------------------------------------------
    # Overall comparison across all methods
    # Uncomment one at a time
    # --------------------------------------------------------

    # plot_single_pair_metric_across_methods_vertical_compact(
    #     df,
    #     group_id=2,
    #     fixed_id=0,
    #     moving_id=1,
    #     metric="spot_topk",
    # )


    plot_overall_metric_across_methods(df, metric="spot_topk")
    # plot_overall_metric_across_methods_vertical_compact(df, metric="spatial_corr")
    # plot_overall_metric_across_methods_vertical_compact(df, metric="center_shift")

    group_id = 2
    # plot_domain_distance_method_lines(df, metric="spot_topk", group_id=group_id)
    plot_domain_distance_method_heatmaps(
        df,
        metric="spot_topk",
        group_id=group_id,
        # format_distance_label=format_distance_label,
        # format_domain_label=format_domain_label,
        xtick_shift=(0.0, 0.05),
        ytick_shift=(0.05, 0.0),
        DISTANCE_ORDER=DISTANCE_ORDER
    )

    # plot_axis_domain_heatmap(df, metric="dice", method="axis", group_id=group_id, mode="distance_bin")
    # plot_axis_domain_heatmap(df, metric="spatial_corr", method="axis", group_id=group_id, mode="distance_bin")
    # plot_axis_domain_heatmap(df, metric="center_shift", method="axis", group_id=group_id, mode="distance_bin")

    # plot_axis_domain_heatmap(df, metric="dice", method="axis", group_id=group_id, mode="distance_um")
    # plot_axis_domain_heatmap(df, metric="spatial_corr", method="axis", group_id=group_id, mode="distance_um")
    # plot_axis_domain_heatmap(df, metric="center_shift", method="axis", group_id=group_id, mode="distance_um")

    # plot_axis_domain_pairtype_heatmap(df, metric="dice", method="gpsa", group_id=group_id)
    # plot_axis_domain_pairtype_heatmap(df, metric="spatial_corr", method="axis", group_id=group_id)
    # plot_axis_domain_pairtype_heatmap(df, metric="center_shift", method="axis", group_id=group_id)