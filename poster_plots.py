import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Style Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

# 1. Load the Data
TSV_PATH = "detector_demographic_bias_results.tsv"
RAW_PATH = "Altered Datasets/ASAP 2.0/pared_dataset.tsv"

if not os.path.exists(TSV_PATH):
    raise FileNotFoundError(
        f"Could not find '{TSV_PATH}'. Please ensure your prior script has run successfully "
        "and generated this output file in the same directory."
    )

df = pd.read_csv(TSV_PATH, sep="\t")

# Load raw data if available for first quick plot
try:
    raw = pd.read_csv(RAW_PATH, sep="\t")
    race_counts = raw["race_ethnicity"].value_counts()
except Exception:
    # Fallback if raw file isn't present in current execution context
    race_counts = {}


# --- QUICK INITIAL PLOT (Fixed column mapping & references) ---
def plot_quick_auc_and_artifacts():
    sub = df[df["Demographic_Feature"] == "race_ethnicity"].sort_values("AUC")
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(sub))

    ax.hlines(y_pos, sub["AUC_CI_Lower"], sub["AUC_CI_Upper"], color="#2a78d6", linewidth=2)
    ax.scatter(sub["AUC"], y_pos, color="#0c447c", zorder=3, s=60)

    # Use dynamically computed counts from raw dataset fallback if column is missing
    yticklabels = []
    for _, row in sub.iterrows():
        subgroup = row['Subgroup']
        if 'Sample_Size_Students' in row and not pd.isna(row['Sample_Size_Students']):
            n_val = int(row['Sample_Size_Students'])
        else:
            n_val = int(race_counts.get(subgroup, 0))
        
        yticklabels.append(f"{subgroup} (N={n_val:,})" if n_val > 0 else f"{subgroup}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("AUC")
    ax.set_title("Detector separability by race/ethnicity")
    plt.tight_layout()
    plt.savefig("auc_by_race.png", dpi=300)
    plt.close()

    # Double subplot for grade level metrics
    grade = df[df["Demographic_Feature"] == "grade_level"].sort_values("Subgroup")
    if len(grade) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        x = range(len(grade))
        axes[0].bar([i - 0.2 for i in x], grade["Human_False_Positive_Rate"], width=0.4, label="FPR", color="#eb6834")
        axes[0].bar([i + 0.2 for i in x], grade["AI_Recall_TPR"], width=0.4, label="Recall", color="#2a78d6")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(grade["Subgroup"])
        axes[0].set_title("Threshold-dependent metrics")
        axes[0].legend()

        axes[1].bar(x, grade["AUC"], color="#0c447c")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(grade["Subgroup"])
        axes[1].set_ylim(0.9, 1.0)
        axes[1].set_title("Threshold-independent metric (AUC)")

        plt.tight_layout()
        plt.savefig("grade_calibration_artifact.png", dpi=300)
        plt.close()


# --- GRAPH 1: The "Grade-Level Slide" (Line Graph with Error Bars) ---
def plot_grade_level_slide(df):
    grade_df = df[df["Demographic_Feature"] == "grade_level"].copy()
    if len(grade_df) == 0:
        return
    
    # Ensure grade level is treated as a sorted numeric or categorical index
    grade_df["Subgroup"] = pd.to_numeric(grade_df["Subgroup"])
    grade_df = grade_df.sort_values("Subgroup")
    
    # Calculate asymmetric error bounds for the CI
    y = grade_df["Human_False_Positive_Rate"]
    yerr_lower = (y - grade_df["Human_FPR_CI_Lower"]).clip(lower=0)
    yerr_upper = (grade_df["Human_FPR_CI_Upper"] - y).clip(lower=0)
    yerr = np.vstack([yerr_lower, yerr_upper])

    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.errorbar(
        grade_df["Subgroup"], y, yerr=yerr, 
        fmt="-o", color="#d95f02", linewidth=2.5, elinewidth=2, 
        capsize=6, capthick=2, markersize=8, label="FPR with 95% CI"
    )
    
    # Formatting
    ax.set_title("False Positive Rate Escalates in Younger Grades", pad=15, weight="bold")
    ax.set_xlabel("Student Grade Level", labelpad=10)
    ax.set_ylabel("False Positive Rate (FPR)", labelpad=10)
    ax.set_xticks(grade_df["Subgroup"])
    ax.set_xticklabels([f"Grade {int(g)}" for g in grade_df["Subgroup"]])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    ax.set_ylim(-0.02, 0.25)
    
    plt.tight_layout()
    plt.savefig("grade_level_fpr_slide.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved: grade_level_fpr_slide.png")


# --- GRAPH 2: Demographic FPR Comparison (Explicit & Robust Alignment) ---
def plot_demographic_fpr_comparison(df):
    demo_features = ["race_ethnicity", "gender", "ell_status", "economically_disadvantaged"]
    plot_df = df[df["Demographic_Feature"].isin(demo_features)].copy()
    if len(plot_df) == 0:
        return
    
    # Clean up names for display
    feature_rename = {
        "race_ethnicity": "Race/Ethnicity",
        "gender": "Gender",
        "ell_status": "ELL Status",
        "economically_disadvantaged": "Socioeconomic Status"
    }
    plot_df["Demographic_Category_Label"] = plot_df["Demographic_Feature"].map(feature_rename)
    
    # Calculate overall baseline FPR to show on the graph safely
    total_fp = (df["Sample_Size_Students"] * df["Human_False_Positive_Rate"]).sum()
    total_students = df["Sample_Size_Students"].sum()
    baseline_fpr = total_fp / total_students if total_students > 0 else df["Human_False_Positive_Rate"].mean()

    # Sort so that bars group nicely by Demographic Feature
    plot_df = plot_df.sort_values(by=["Demographic_Category_Label", "Subgroup"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Render main bar chart using explicit hue-mapping to bypass alignment bugs
    sns.barplot(
        data=plot_df,
        y="Subgroup",
        x="Human_False_Positive_Rate",
        hue="Demographic_Category_Label",
        dodge=False,
        palette="Set2",
        ax=ax
    )
    
    # Draw error bars perfectly relative to actual visual bar centers
    # This avoids index mismatch errors common in raw patch access
    for i, row in plot_df.iterrows():
        x_val = row["Human_False_Positive_Rate"]
        x_lower = row["Human_FPR_CI_Lower"]
        x_upper = row["Human_FPR_CI_Upper"]
        
        # Clip error limits dynamically so bounds stay mathematically sound
        err_left = max(0, x_val - x_lower)
        err_right = max(0, x_upper - x_val)
        
        # Grab coordinate of corresponding drawn bar patch directly
        bar_patch = ax.patches[i]
        y_center = bar_patch.get_y() + bar_patch.get_height() / 2.0
        
        ax.errorbar(
            x=[x_val], y=[y_center], 
            xerr=[[err_left], [err_right]],
            fmt='none', color='black', capsize=4, elinewidth=1.5, capthick=1.5
        )

    # Add global baseline average line
    ax.axvline(
        baseline_fpr, color="red", linestyle="--", alpha=0.7, 
        label=f"Global Average FPR ({baseline_fpr*100:.1f}%)"
    )

    # Formatting
    ax.set_title("False Positive Rate Across Student Demographics", pad=15, weight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", labelpad=10)
    ax.set_ylabel("Student Subgroup", labelpad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    ax.legend(title="Category Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.savefig("demographic_fpr_comparison.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved: demographic_fpr_comparison.png")


# --- GRAPH 3: The Recall vs. FPR Trade-off (Scatter Plot) ---
def plot_recall_fpr_tradeoff(df):
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Define custom color palette
    categories = df["Demographic_Feature"].unique()
    colors = sns.color_palette("bright", len(categories))
    color_map = dict(zip(categories, colors))
    
    # Plot each point
    for category in categories:
        sub_df = df[df["Demographic_Feature"] == category]
        ax.scatter(
            sub_df["Human_False_Positive_Rate"], 
            sub_df["AI_Recall_TPR"], 
            color=color_map[category], 
            s=120, 
            edgecolor="black", 
            linewidth=1, 
            label=category.replace("_", " ").title(),
            alpha=0.85
        )
        
        # Add labels for individual subgroups
        for _, row in sub_df.iterrows():
            ax.annotate(
                row["Subgroup"], 
                (row["Human_False_Positive_Rate"], row["AI_Recall_TPR"]),
                textcoords="offset points", 
                xytext=(0, 8), 
                ha="center", 
                fontsize=8, 
                weight="semibold"
            )

    # Target Goal Indicator (Ideal spot)
    ax.scatter([0.0], [1.0], color="gold", marker="*", s=300, edgecolor="black", label="Ideal Detector Target")
    
    # Formatting
    ax.set_title("Detector Trade-Off: AI Catch Rate vs. False Positives", pad=15, weight="bold")
    ax.set_xlabel("False Positive Rate (FPR) - Lower is Better", labelpad=10)
    ax.set_ylabel("AI Recall (TPR) - Higher is Better", labelpad=10)
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(0.91, 1.01)
    
    ax.legend(loc="lower right", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig("recall_fpr_tradeoff_scatter.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved: recall_fpr_tradeoff_scatter.png")


# --- Execution ---
if __name__ == "__main__":
    print("Generating visual assets...")
    plot_quick_auc_and_artifacts()
    plot_grade_level_slide(df)
    plot_demographic_fpr_comparison(df)
    plot_recall_fpr_tradeoff(df)
    print("All graphics created successfully!")