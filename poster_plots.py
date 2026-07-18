import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Poster-scale fonts: these are large enough to read from several feet away.
# If your poster template uses a different base size, scale everything here
# proportionally rather than tweaking individual plots.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 20,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "legend.title_fontsize": 18,
    "figure.titlesize": 28
})

VALUE_LABEL_SIZE = 16   # font size for the printed number/percentage on each bar/point
N_LABEL_SIZE = 15       # font size for sample-size text where it's separate from tick labels

# --- Figure sizes (width, height) in inches -- tweak these directly to make ---
# --- any individual graph wider/taller. Width is the first number.        ---
FIGSIZE_AUC_RACE = (17, 7)              # race/ethnicity AUC forest plot
FIGSIZE_GRADE_ARTIFACT = (20, 7.5)      # two-panel FPR/Recall vs AUC by grade
FIGSIZE_GRADE_RECALL_AUC = (18, 7.5)    # Recall + AUC bars by grade
FIGSIZE_FPR_COMPARISON = (20, 10)        # FPR across all demographics
FIGSIZE_SCATTER = (13, 11)              # Recall vs FPR trade-off scatter

#Load Data
TSV_PATH = "detector_demographic_bias_results.tsv"
RAW_PATH = "Altered Datasets/ASAP 2.0/pared_dataset.tsv"

if not os.path.exists(TSV_PATH):
    raise FileNotFoundError(
        f"Could not find '{TSV_PATH}'. Please ensure your prior script has run successfully "
        "and generated this output file in the same directory."
    )

df = pd.read_csv(TSV_PATH, sep="\t")

#Load from dataset
try:
    raw = pd.read_csv(RAW_PATH, sep="\t")
    race_counts = raw["race_ethnicity"].value_counts()
except Exception:
    race_counts = {}


def _n_label(n_val: int) -> str:
    """Consistent 'N=1,234' formatting used across every plot."""
    return f"N={n_val:,}"


def _grade_level_df(df):
    grade_df = df[df["Demographic_Feature"] == "grade_level"].copy()
    grade_df["Subgroup"] = pd.to_numeric(grade_df["Subgroup"])
    return grade_df.sort_values("Subgroup").reset_index(drop=True)


def plot_quick_auc_and_artifacts():
    sub = df[df["Demographic_Feature"] == "race_ethnicity"].sort_values("AUC")
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_AUC_RACE)
    y_pos = range(len(sub))

    ax.hlines(y_pos, sub["AUC_CI_Lower"], sub["AUC_CI_Upper"], color="#2a78d6", linewidth=3)
    ax.scatter(sub["AUC"], y_pos, color="#0c447c", zorder=3, s=140)

    yticklabels = []
    for _, row in sub.iterrows():
        subgroup = row['Subgroup']
        if 'Sample_Size_Students' in row and not pd.isna(row['Sample_Size_Students']):
            n_val = int(row['Sample_Size_Students'])
        else:
            n_val = int(race_counts.get(subgroup, 0))
        yticklabels.append(f"{subgroup} ({_n_label(n_val)})" if n_val > 0 else f"{subgroup}")

    # Value label: the AUC point estimate itself, printed just above each dot
    for y, (_, row) in zip(y_pos, sub.iterrows()):
        ax.annotate(
            f"{row['AUC']:.3f}",
            (row["AUC"], y),
            textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=VALUE_LABEL_SIZE, weight="bold", color="#0c447c"
        )

    # Extra headroom above the top row so its value-label annotation (offset
    # 14pt above the dot) never collides with the title text.
    ax.set_ylim(-0.8, len(sub) - 1 + 0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("AUC")
    ax.set_title("Detector separability by race/ethnicity", pad=28)
    plt.tight_layout()
    plt.savefig("auc_by_race.png", dpi=300)
    plt.close()

    grade = _grade_level_df(df)
    if len(grade) > 0:
        grade_xticklabels = [
            f"Grade {int(row['Subgroup'])}\n({_n_label(int(row['Sample_Size_Students']))})"
            for _, row in grade.iterrows()
        ]

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_GRADE_ARTIFACT)
        x = range(len(grade))

        bars_fpr = axes[0].bar([i - 0.2 for i in x], grade["Human_False_Positive_Rate"],
                                width=0.4, label="FPR", color="#eb6834")
        bars_recall = axes[0].bar([i + 0.2 for i in x], grade["AI_Recall_TPR"],
                                   width=0.4, label="Recall", color="#2a78d6")
        # Value labels: the actual percentage above each bar
        for b in bars_fpr:
            axes[0].annotate(f"{b.get_height()*100:.1f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                              textcoords="offset points", xytext=(0, 6), ha="center",
                              fontsize=VALUE_LABEL_SIZE, weight="bold")
        for b in bars_recall:
            axes[0].annotate(f"{b.get_height()*100:.1f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                              textcoords="offset points", xytext=(0, 6), ha="center",
                              fontsize=VALUE_LABEL_SIZE, weight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(grade_xticklabels)
        axes[0].set_ylim(0, 1.12)
        axes[0].set_title("Threshold-dependent metrics", pad=20)
        axes[0].legend()

        bars_auc = axes[1].bar(x, grade["AUC"], color="#0c447c")
        for b in bars_auc:
            axes[1].annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                              textcoords="offset points", xytext=(0, 6), ha="center",
                              fontsize=VALUE_LABEL_SIZE, weight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(grade_xticklabels)
        axes[1].set_ylim(0.9, 1.03)
        axes[1].set_title("Threshold-independent metric (AUC)", pad=20)

        plt.tight_layout()
        plt.savefig("grade_calibration_artifact.png", dpi=300)
        plt.close()


# Recall + AUC together from grade level
def plot_grade_level_recall_auc(df):
    grade_df = _grade_level_df(df)
    if len(grade_df) == 0:
        return

    x = np.arange(len(grade_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_GRADE_RECALL_AUC)

    # Recall bars + asymmetric 95% CI whiskers
    recall = grade_df["AI_Recall_TPR"]
    recall_err = np.vstack([
        (recall - grade_df["AI_Recall_CI_Lower"]).clip(lower=0),
        (grade_df["AI_Recall_CI_Upper"] - recall).clip(lower=0),
    ])
    bars_recall = ax.bar(
        x - width / 2, recall, width,
        label="Recall (AI catch rate)", color="#2a78d6",
        yerr=recall_err, capsize=6, error_kw={"elinewidth": 2},
    )

    # AUC bars + 95% CI whiskers
    auc = grade_df["AUC"]
    auc_err = np.vstack([
        (auc - grade_df["AUC_CI_Lower"]).clip(lower=0),
        (grade_df["AUC_CI_Upper"] - auc).clip(lower=0),
    ])
    bars_auc = ax.bar(
        x + width / 2, auc, width,
        label="AUC (threshold-free)", color="#6a3d9a",
        yerr=auc_err, capsize=6, error_kw={"elinewidth": 2},
    )

    # Value labels: printed above each bar's upper CI whisker, not the bar
    # top itself, so the text never collides with the error cap.
    for b, err_up in zip(bars_recall, recall_err[1]):
        ax.annotate(f"{b.get_height()*100:.1f}%",
                    (b.get_x() + b.get_width()/2, b.get_height() + err_up),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=VALUE_LABEL_SIZE, weight="bold")
    for b, err_up in zip(bars_auc, auc_err[1]):
        ax.annotate(f"{b.get_height():.3f}",
                    (b.get_x() + b.get_width()/2, b.get_height() + err_up),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=VALUE_LABEL_SIZE, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Grade {int(row['Subgroup'])}\n({_n_label(int(row['Sample_Size_Students']))})"
        for _, row in grade_df.iterrows()
    ])
    ax.set_ylim(0.90, 1.04)
    ax.set_ylabel("Score")
    ax.set_title("Recall and AUC by Grade Level", pad=15, weight="bold")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("grade_level_recall_auc.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved: grade_level_recall_auc.png")


# Demographic FPR Comparison (now includes grade level and disability status)
def plot_demographic_fpr_comparison(df):
    demo_features = [
        "race_ethnicity", "gender", "ell_status",
        "economically_disadvantaged", "student_disability_status", "grade_level"
    ]
    plot_df = df[df["Demographic_Feature"].isin(demo_features)].copy()
    if len(plot_df) == 0:
        return

    feature_rename = {
        "race_ethnicity": "Race/Ethnicity",
        "gender": "Gender",
        "ell_status": "ELL Status",
        "economically_disadvantaged": "Socioeconomic Status",
        "student_disability_status": "Disability Status",
        "grade_level": "Grade Level",
    }
    plot_df["Demographic_Category_Label"] = plot_df["Demographic_Feature"].map(feature_rename)

    is_grade = plot_df["Demographic_Feature"] == "grade_level"
    plot_df.loc[is_grade, "Subgroup"] = plot_df.loc[is_grade, "Subgroup"].apply(
        lambda g: f"Grade {int(g)}"
    )

    def _sort_key(row):
        if row["Demographic_Feature"] == "grade_level":
            grade_num = int(row["Subgroup"].replace("Grade ", ""))
            return f"{grade_num:02d}"
        return str(row["Subgroup"])

    plot_df["_sort_key"] = plot_df.apply(_sort_key, axis=1)
    plot_df = plot_df.sort_values(by=["Demographic_Category_Label", "_sort_key"]).reset_index(drop=True)

    plot_df["Subgroup_Label"] = plot_df.apply(
        lambda row: f"{row['Subgroup']} ({_n_label(int(row['Sample_Size_Students']))})",
        axis=1
    )

    total_fp = (df["Sample_Size_Students"] * df["Human_False_Positive_Rate"]).sum()
    total_students = df["Sample_Size_Students"].sum()
    baseline_fpr = total_fp / total_students if total_students > 0 else df["Human_False_Positive_Rate"].mean()

    fig, ax = plt.subplots(figsize=FIGSIZE_FPR_COMPARISON)

    sns.barplot(
        data=plot_df,
        y="Subgroup_Label",
        x="Human_False_Positive_Rate",
        hue="Demographic_Category_Label",
        dodge=False,
        palette="Set2",
        ax=ax
    )

    # Compute a consistent offset for the value labels, based on the overall
    # x-range, so labels sit clear of the error whiskers no matter the scale.
    x_axis_span = plot_df["Human_FPR_CI_Upper"].max()
    label_offset = x_axis_span * 0.02

    for i, row in plot_df.iterrows():
        x_val = row["Human_False_Positive_Rate"]
        x_lower = row["Human_FPR_CI_Lower"]
        x_upper = row["Human_FPR_CI_Upper"]

        err_left = max(0, x_val - x_lower)
        err_right = max(0, x_upper - x_val)

        bar_patch = ax.patches[i]
        y_center = bar_patch.get_y() + bar_patch.get_height() / 2.0

        ax.errorbar(
            x=[x_val], y=[y_center],
            xerr=[[err_left], [err_right]],
            fmt='none', color='black', capsize=5, elinewidth=2, capthick=2
        )

        # Value label: the FPR percentage, printed just past the upper CI
        # whisker so it never overlaps the bar or the error cap.
        ax.annotate(
            f"{x_val*100:.1f}%",
            (x_upper + label_offset, y_center),
            ha="left", va="center", fontsize=VALUE_LABEL_SIZE, weight="bold"
        )

    ax.axvline(
        baseline_fpr, color="red", linestyle="--", alpha=0.7,
        label=f"Global Average FPR ({baseline_fpr*100:.1f}%)"
    )

    # Give the value labels room on the right edge of the plot
    ax.set_xlim(right=plot_df["Human_FPR_CI_Upper"].max() * 1.18)

    ax.set_title("False Positive Rate Across Student Demographics", pad=15, weight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", labelpad=10)
    ax.set_ylabel("Student Subgroup", labelpad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    ax.legend(title="Category Group", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("demographic_fpr_comparison.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved: demographic_fpr_comparison.png")


# --- GRAPH 3: The Recall vs. FPR Trade-off (Scatter Plot) ---
def plot_recall_fpr_tradeoff(df):
    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)

    categories = df["Demographic_Feature"].unique()
    colors = sns.color_palette("bright", len(categories))
    color_map = dict(zip(categories, colors))

    # Points that sit close together in FPR/Recall space produce overlapping
    # labels if every label is offset the same way. Sort all points by x
    # position and alternate the vertical offset (and left/right alignment
    # for the closest neighbors) so adjacent labels stagger instead of
    # stacking directly on top of each other.
    all_rows = df.sort_values("Human_False_Positive_Rate").reset_index(drop=True)
    offsets = [(0, 14), (0, 34), (0, 54)]  # cycle through 3 vertical heights

    for category in categories:
        sub_df = df[df["Demographic_Feature"] == category]
        ax.scatter(
            sub_df["Human_False_Positive_Rate"],
            sub_df["AI_Recall_TPR"],
            color=color_map[category],
            s=180,
            edgecolor="black",
            linewidth=1.5,
            label=category.replace("_", " ").title(),
            alpha=0.85
        )

    # Label pass done separately (across ALL points at once, not per-category)
    # so the staggering cycle is based on overall proximity, not reset for
    # each color group.
    for i, row in all_rows.iterrows():
        n_val = int(row["Sample_Size_Students"]) if not pd.isna(row["Sample_Size_Students"]) else None
        n_part = f" ({_n_label(n_val)})" if n_val else ""
        label_text = (
            f"{row['Subgroup']}{n_part}\n"
            f"FPR {row['Human_False_Positive_Rate']*100:.1f}% / "
            f"Recall {row['AI_Recall_TPR']*100:.1f}%"
        )
        xoff, yoff = offsets[i % len(offsets)]
        ax.annotate(
            label_text,
            (row["Human_False_Positive_Rate"], row["AI_Recall_TPR"]),
            textcoords="offset points",
            xytext=(xoff, yoff),
            ha="center",
            fontsize=12,
            weight="semibold"
        )

    ax.scatter([0.0], [1.0], color="gold", marker="*", s=500, edgecolor="black", label="Ideal Detector Target")

    ax.set_title("Detector Trade-Off: AI Catch Rate vs. False Positives", pad=15, weight="bold")
    ax.set_xlabel("False Positive Rate (FPR) - Lower is Better", labelpad=10)
    ax.set_ylabel("AI Recall (TPR) - Higher is Better", labelpad=10)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))

    ax.set_xlim(-0.01, 0.24)
    ax.set_ylim(0.90, 1.03)

    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=13)

    plt.tight_layout()
    plt.savefig("recall_fpr_tradeoff_scatter.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved: recall_fpr_tradeoff_scatter.png")


# --- Execution ---
if __name__ == "__main__":
    print("Generating visual assets...")
    plot_quick_auc_and_artifacts()
    plot_grade_level_recall_auc(df)
    plot_demographic_fpr_comparison(df)
    plot_recall_fpr_tradeoff(df)
    print("All graphics created successfully!")