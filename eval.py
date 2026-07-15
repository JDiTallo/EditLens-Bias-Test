import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

sys.path.insert(0, "editlens_src/scripts/eval")

from editlens_src.scripts.eval.threshold import find_optimal_threshold

# Load the dataset
df = pd.read_csv("Altered Datasets/ASAP 2.0/pared_dataset.tsv", sep="\t")

# Scores of originally written papers with AI written ones
predictions = np.concatenate([df['score_original'].values, df['score_light'].values])

#Labels human written and AI text (0 and 1, repspectively)
labels = np.concatenate([np.zeros(len(df)), np.ones(len(df))])

optimal_threshold, max_f1 = find_optimal_threshold(predictions, labels)

# List of demographic groups to check
demographics = [
    'race_ethnicity', 
    'gender', 
    'ell_status', 
    'economically_disadvantaged', 
    'student_disability_status', 
    'grade_level'
]


demographics_dict = {}

for feature in demographics:
    unique_items = df[feature].dropna().unique().tolist()
    demographics_dict[feature] = unique_items

def calculate_classification_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics (with guards against division-by-zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Also known as True Positive Rate / Sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0     # False Positive Rate
    
    # F1-Score is the harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Overall Accuracy
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1-Score": f1_score,
        "False Positive Rate": fpr
    }

def eval(df, threshold):
    humans_true = np.zeros(len(df))
    ai_true = np.ones(len(df))

    humans_pred = (df['score_original'] >= threshold).astype(int).values
    ai_pred = (df['score_light'] >= threshold).astype(int).values

    human_result = calculate_classification_metrics(humans_true, humans_pred)
    ai_result = calculate_classification_metrics(ai_true, ai_pred)


    y_true = np.concatenate([humans_true, ai_true])
    y_pred = np.concatenate([humans_pred, ai_pred])
    combined_results = calculate_classification_metrics(y_true, y_pred)

    return human_result, ai_result, combined_results

def print_result(results, title="Performance"):
    print(title)
    for metric, score in results.items():
        if metric == "F1-Score":
            print(f"{metric}: {score:.4f}  <-- (Balanced measure of model quality)")
        else:
            print(f"{metric}: {score:.4f}")

human_result, ai_result, combined_results = eval(df, optimal_threshold)

# print_result(human_result, "Human Performance")
# print_result(ai_result, "AI performance")
# print_result(combined_results, "Combined Results")





