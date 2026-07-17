import matplotlib.pyplot as plt
import pandas as pd

# Load both datasets up front
df = pd.read_csv("detector_demographic_bias_results.tsv", sep="\t")
raw = pd.read_csv("Altered Datasets/ASAP 2.0/pared_dataset.tsv", sep="\t")

# 1. Calculate sample sizes (N) for each race/ethnicity from the raw data
race_counts = raw["race_ethnicity"].value_counts()

sub = df[df["Demographic_Feature"] == "race_ethnicity"].sort_values("AUC")

fig, ax = plt.subplots(figsize=(8, 5)) # Slightly wider to accommodate longer labels
y_pos = range(len(sub))

# The CI as a horizontal line
ax.hlines(y_pos, sub["AUC_CI_Lower"], sub["AUC_CI_Upper"], color="#2a78d6", linewidth=2)
# The point estimate as a dot on top
ax.scatter(sub["AUC"], y_pos, color="#0c447c", zorder=3, s=60)

# 2. Construct dynamic labels with sample sizes: "Subgroup (N=X,XXX)"
# Use this if you have a "Sample_Size" column in df
yticklabels = [
    f"{row['Subgroup']} (N={int(row['Sample_Size_Students']):,})" 
    for _, row in sub.iterrows()
]

ax.set_yticks(y_pos)
ax.set_yticklabels(yticklabels) # Set the new labels
ax.set_xlabel("AUC")
ax.set_title("Detector separability by race/ethnicity")
plt.tight_layout()
plt.savefig("auc_by_race.png", dpi=300)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

grade = df[df["Demographic_Feature"] == "grade_level"].sort_values("Subgroup")

# Left: FPR and Recall
x = range(len(grade))
axes[0].bar([i - 0.2 for i in x], grade["Human_False_Positive_Rate"], width=0.4, label="FPR", color="#eb6834")
axes[0].bar([i + 0.2 for i in x], grade["AI_Recall_TPR"], width=0.4, label="Recall", color="#2a78d6")
axes[0].set_xticks(x); axes[0].set_xticklabels(grade["Subgroup"])
axes[0].set_title("Threshold-dependent metrics"); axes[0].legend()

# Right: AUC
axes[1].bar(x, grade["AUC"], color="#0c447c")
axes[1].set_xticks(x); axes[1].set_xticklabels(grade["Subgroup"])
axes[1].set_ylim(0.9, 1.0)  # zoom in so the flatness is visible
axes[1].set_title("Threshold-independent metric (AUC)")

plt.tight_layout()
plt.savefig("grade_calibration_artifact.png", dpi=300)







import os
import pandas as pd
import numpy as np
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

if not os.path.exists(TSV_PATH):
    raise FileNotFoundError(
        f"Could not find '{TSV_PATH}'. Please ensure your prior script has run successfully "
        "and generated this output file in the same directory."
    )

df = pd.read_csv(TSV_PATH, sep="\t")


# --- GRAPH 1: The "Grade-Level Slide" (Line Graph with Error Bars) ---
def plot_grade_level_slide(df):
    grade_df = df[df["Demographic_Feature"] == "grade_level"].copy()
    
    # Ensure grade level is treated as a sorted numeric or categorical index
    grade_df["Subgroup"] = pd.to_numeric(grade_df["Subgroup"])
    grade_df = grade_df.sort_values("Subgroup")
    
    # Calculate asymmetric error bounds for the CI
    # Matplotlib errorbars require: [y - lower_bound, upper_bound - y]
    y = grade_df["Human_False_Positive_Rate"]
    yerr_lower = y - grade_df["Human_FPR_CI_Lower"]
    yerr_upper = grade_df["Human_FPR_CI_Upper"] - y
    yerr = np.vstack([yerr_lower, yerr_upper])

    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot line and error bars
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


# --- GRAPH 2: Demographic FPR Comparison (Grouped Bar Chart) ---
def plot_demographic_fpr_comparison(df):
    # Filter out 'grade_level' to focus purely on socio-demographic features
    demo_features = ["race_ethnicity", "gender", "ell_status", "economically_disadvantaged"]
    plot_df = df[df["Demographic_Feature"].isin(demo_features)].copy()
    
    # Clean up names for display
    feature_rename = {
        "race_ethnicity": "Race/Ethnicity",
        "gender": "Gender",
        "ell_status": "ELL Status",
        "economically_disadvantaged": "Socioeconomic Status"
    }
    plot_df["Demographic_Feature"] = plot_df["Demographic_Feature"].map(feature_rename)
    
    # Calculate overall baseline FPR to show on the graph
    total_fp = (df["Sample_Size_Students"] * df["Human_False_Positive_Rate"]).sum()
    total_students = df["Sample_Size_Students"].sum()
    baseline_fpr = total_fp / total_students

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Render main bar chart
    sns.barplot(
        data=plot_df,
        y="Subgroup",
        x="Human_False_Positive_Rate",
        hue="Demographic_Feature",
        dodge=False,
        palette="Set2",
        ax=ax
    )
    
    # Add custom error bars manually to the bars
    # Extract patches (bars) and map them to the dataframe rows
    plot_df = plot_df.reset_index(drop=True)
    for i, bar in enumerate(ax.patches):
        if i >= len(plot_df):
            break
        row = plot_df.iloc[i]
        
        # Calculate error limits
        x_val = row["Human_False_Positive_Rate"]
        x_lower = row["Human_FPR_CI_Lower"]
        x_upper = row["Human_FPR_CI_Upper"]
        
        # Draw error bar on top of the bar
        y_pos = bar.get_y() + bar.get_height() / 2.0
        ax.errorbar(
            x=[x_val], y=[y_pos], 
            xerr=[[x_val - x_lower], [x_upper - x_val]],
            fmt='none', color='black', capsize=4, elinewidth=1.5, capthick=1.5
        )

    # Add a global baseline average line
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
    
    # Define a custom color palette for the categories
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

    # Target Goal Indicator (Upper-Left Corner represents the Perfect Detector)
    ax.scatter([0.0], [1.0], color="gold", marker="*", s=300, edgecolor="black", label="Ideal Detector Target")
    
    # Formatting
    ax.set_title("Detector Trade-Off: AI Catch Rate vs. False Positives", pad=15, weight="bold")
    ax.set_xlabel("False Positive Rate (FPR) - Lower is Better", labelpad=10)
    ax.set_ylabel("AI Recall (TPR) - Higher is Better", labelpad=10)
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
    
    # Set generous limits to display annotations clearly
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(0.91, 1.01)
    
    ax.legend(loc="lower right", frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig("recall_fpr_tradeoff_scatter.png", dpi=300)
    plt.close()
    print("[SUCCESS] Saved: recall_fpr_tradeoff_scatter.png")


# --- Execution Execution ---
if __name__ == "__main__":
    print("Generating visual assets for poster...")
    plot_grade_level_slide(df)
    plot_demographic_fpr_comparison(df)
    plot_recall_fpr_tradeoff(df)
    print("All poster graphics created successfully!")