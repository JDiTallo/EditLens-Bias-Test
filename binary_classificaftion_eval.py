import pandas as pd
import numpy as np
import os
import sys

# Ensure custom import path is included
sys.path.insert(0, "editlens_src/scripts/eval")
from editlens_src.scripts.eval.threshold import find_optimal_threshold

DATASET_PATH = "Altered Datasets/ASAP 2.0/pared_dataset.tsv"
OUTPUT_TSV_PATH = "detector_demographic_bias_results.tsv"

# The columns you want to evaluate for bias
DEMOGRAPHIC_COLUMNS = [
    'race_ethnicity', 
    'gender', 
    'ell_status', 
    'economically_disadvantaged', 
    'student_disability_status', 
    'grade_level'
]

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
        "False Positive Rate": fpr
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
            
            # Print console metrics dynamically
            print(f"\nSubgroup: {category} (Sample Size: {len(sub_df)} students / {total_essays} essays)")
            print(f"  [Human Essays (Innocence Check)]")
            print(f"    - False Positive Rate:      {format_metric(human_res['False Positive Rate'])}")
            print(f"    - Correct Human Passing Rate: {format_metric(human_res['Accuracy'])}")
            print(f"  [AI-Edited Essays (Catch Rate Check)]")
            print(f"    - Recall (True Positive Rate): {format_metric(ai_res['Recall (TPR)'])}")
            print(f"    - Correct AI Catch Rate:       {format_metric(ai_res['Accuracy'])}")
            print(f"  [Combined Performance (Classroom Simulation)]")
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
                "Correct_Human_Passing_Rate": human_res['Accuracy'],
                "AI_Recall_TPR": ai_res['Recall (TPR)'],
                "Correct_AI_Catch_Rate": ai_res['Accuracy'],
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