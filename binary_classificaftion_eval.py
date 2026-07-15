import pandas as pd
import numpy as np
import os
import sys
import math

from sklearn.metrics import roc_auc_score

# Ensure custom import path is included
sys.path.insert(0, "editlens_src/scripts/eval")
from editlens_src.scripts.eval.threshold import find_optimal_threshold

from statsmodels.stats.proportion import proportion_confint

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple:
    if n == 0:
        return (np.nan, np.nan)
    lower, upper = proportion_confint(count=successes, nobs=n, alpha=alpha, method="wilson")
    return (lower, upper)


DATASET_PATH = "Altered Datasets/ASAP 2.0/pared_dataset.tsv"
OUTPUT_TSV_PATH = "detector_demographic_bias_results.tsv"

# AUC confidence interval metrics
N_BOOTSTRAP = 200
RANDOM_SEED = 42


DEMOGRAPHIC_COLUMNS = [
    'race_ethnicity', 
    'gender', 
    'ell_status', 
    'economically_disadvantaged', 
    'student_disability_status', 
    'grade_level'
]

def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if np.sum(y_true == 1) == 0 or np.sum(y_true == 0) == 0:
        return np.nan
    return float(roc_auc_score(y_true, y_scores))


def bootstrap_auc_ci(sub_df: pd.DataFrame, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
                      seed: int = RANDOM_SEED) -> tuple:
    """
    95% CI for AUC via bootstrap resampling. Resamples STUDENTS (not essays)
    with replacement, so each resampled student's original + light-rewrite
    pair stays together -- this preserves the paired structure of your data
    instead of pretending the two essays are independent draws.
    """
    n = len(sub_df)
    if n < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    orig = sub_df["score_original"].values
    light = sub_df["score_light"].values
    y_true = np.concatenate([np.zeros(n), np.ones(n)])

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_scores = np.concatenate([orig[idx], light[idx]])
        auc = compute_auc(y_true, y_scores)
        if not np.isnan(auc):
            aucs.append(auc)

    if not aucs:
        return (np.nan, np.nan)
    lower = float(np.percentile(aucs, 100 * alpha / 2))
    upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return (lower, upper)

def load_data(file_path: str, sep="\t") -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find dataset at: {file_path}")
    return pd.read_csv(file_path, sep=sep)

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else -1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else -1.0  
    fpr = fp / (fp + tn) if (fp + tn) > 0 else -1.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else -1.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1-Score": f1_score,
        "False Positive Rate": fpr,

        "_tp": int(tp),
        "_fp": int(fp),
        "_tn": int(tn),
        "_fn": int(fn),
    }


def evaluate_threshold(df: pd.DataFrame, threshold: float) -> tuple:
    humans_true = np.zeros(len(df))
    ai_true = np.ones(len(df))

    # Convert continuous model outputs to binary predictions
    humans_pred = (df['score_original'] >= threshold).astype(int).values
    ai_pred = (df['score_light'] >= threshold).astype(int).values

    human_result = calculate_classification_metrics(humans_true, humans_pred)
    ai_result = calculate_classification_metrics(ai_true, ai_pred)

    y_true = np.concatenate([humans_true, ai_true])
    y_pred = np.concatenate([humans_pred, ai_pred])
    combined_results = calculate_classification_metrics(y_true, y_pred)

    return human_result, ai_result, combined_results


def format_metric(val: float) -> str:
    #Formats -1.0 values as "N/A" for clarity
    if np.isclose(val, -1.0) or val < 0:
        return "N/A"
    return f"{val:.4f}"


def format_ci(lower: float, upper: float) -> str:
    if np.isnan(lower) or np.isnan(upper):
        return "N/A"
    return f"[{lower:.4f}, {upper:.4f}]"


def main():
    print("--- Starting Demographic Bias Pipeline ---")
    
    # Load and optimize
    df = load_data(DATASET_PATH)
    print(f"Dataset successfully loaded. Total Students: {len(df)}")
    
    predictions = np.concatenate([df['score_original'].values, df['score_light'].values])
    labels = np.concatenate([np.zeros(len(df)), np.ones(len(df))])
    
    optimal_threshold, max_f1 = find_optimal_threshold(predictions, labels)
    print(f"Optimal Threshold Found: {optimal_threshold:.4f} (Max F1: {max_f1:.4f})\n")
    
    # Gather unique categories for each target demographic variable
    demographics_dict = {
        feature: df[feature].dropna().unique().tolist()
        for feature in DEMOGRAPHIC_COLUMNS if feature in df.columns
    }
    
    export_data = []

    # Loop and analyze
    for feature, categories in demographics_dict.items():
        print(f"\nEvaluating Category Group: {feature.upper()}")
        print("=" * 50)
        
        for category in categories:
            sub_df = df[df[feature] == category]
            
            if len(sub_df) == 0:
                continue
                
            human_res, ai_res, comb_res = evaluate_threshold(sub_df, optimal_threshold)
            total_essays = len(sub_df) * 2

            # --- Confidence intervals ---
            # FPR is computed over all human-written essays in this subgroup:
            # denominator = fp + tn = len(sub_df) (every row contributes one
            # human essay with true label 0). Successes = essays wrongly
            # flagged as AI (fp).
            n_human = human_res["_fp"] + human_res["_tn"]
            fpr_lower, fpr_upper = wilson_ci(human_res["_fp"], n_human)

            # Recall is computed over all AI-edited essays in this subgroup:
            # denominator = tp + fn = len(sub_df). Successes = essays
            # correctly flagged as AI (tp).
            n_ai = ai_res["_tp"] + ai_res["_fn"]
            recall_lower, recall_upper = wilson_ci(ai_res["_tp"], n_ai)

            # Precision is computed over ALL essays flagged as AI, pooled
            # across both the human and AI-edited arms (comb_res already does
            # this pooling). Denominator = tp + fp = every essay the detector
            # called "AI", regardless of which arm it actually came from.
            # Successes = the ones it got right (tp).
            n_flagged = comb_res["_tp"] + comb_res["_fp"]
            precision_lower, precision_upper = wilson_ci(comb_res["_tp"], n_flagged)

            # AUC: threshold-independent measure of how well the detector
            # separates this subgroup's human essays from its AI-edited
            # essays, using the raw continuous scores (not the thresholded
            # predictions used above).
            auc_y_true = np.concatenate([np.zeros(len(sub_df)), np.ones(len(sub_df))])
            auc_y_scores = np.concatenate([sub_df['score_original'].values, sub_df['score_light'].values])
            auc = compute_auc(auc_y_true, auc_y_scores)
            auc_lower, auc_upper = bootstrap_auc_ci(sub_df)

            # Print console metrics dynamically
            print(f"\nSubgroup: {category} (Sample Size: {len(sub_df)} students / {total_essays} essays)")
            print(f"  [Human Essays (Innocence Check)]")
            print(f"    - False Positive Rate:      {format_metric(human_res['False Positive Rate'])}  95% CI: {format_ci(fpr_lower, fpr_upper)}")
            print(f"    - Correct Human Passing Rate: {format_metric(human_res['Accuracy'])}")
            print(f"  [AI-Edited Essays (Catch Rate Check)]")
            print(f"    - Recall (True Positive Rate): {format_metric(ai_res['Recall (TPR)'])}  95% CI: {format_ci(recall_lower, recall_upper)}")
            print(f"    - Correct AI Catch Rate:       {format_metric(ai_res['Accuracy'])}")
            print(f"  [Combined Performance (Classroom Simulation)]")
            print(f"    - Precision:                {format_metric(comb_res['Precision'])}  95% CI: {format_ci(precision_lower, precision_upper)}")
            print(f"    - AUC (threshold-free):     {format_metric(auc)}  95% CI: {format_ci(auc_lower, auc_upper)}")
            print(f"    - Combined F1-Score:        {format_metric(comb_res['F1-Score'])}")
            print(f"    - Combined Accuracy:        {format_metric(comb_res['Accuracy'])}")
            print("-" * 50)

            # Package records for output
            export_data.append({
                "Demographic_Feature": feature,
                "Subgroup": category,
                "Sample_Size_Students": len(sub_df),
                "Total_Essays_Tested": total_essays,
                "Human_False_Positive_Rate": human_res['False Positive Rate'],
                "Human_FPR_CI_Lower": fpr_lower,
                "Human_FPR_CI_Upper": fpr_upper,
                "Correct_Human_Passing_Rate": human_res['Accuracy'],
                "AI_Recall_TPR": ai_res['Recall (TPR)'],
                "AI_Recall_CI_Lower": recall_lower,
                "AI_Recall_CI_Upper": recall_upper,
                "Correct_AI_Catch_Rate": ai_res['Accuracy'],
                "Precision": comb_res['Precision'],
                "Precision_CI_Lower": precision_lower,
                "Precision_CI_Upper": precision_upper,
                "AUC": auc,
                "AUC_CI_Lower": auc_lower,
                "AUC_CI_Upper": auc_upper,
                "Combined_F1_Score": comb_res['F1-Score'],
                "Combined_Accuracy": comb_res['Accuracy']
            })

    # Export structured results
    if export_data:
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(OUTPUT_TSV_PATH, sep='\t', index=False)
        print(f"\n[SUCCESS] Exported all bias evaluations to: '{OUTPUT_TSV_PATH}'")
    else:
        print("\n[WARNING] No demographic data processed. TSV was not written.")


if __name__ == "__main__":
    main()