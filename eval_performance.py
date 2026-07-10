import pandas as pd
import statistics

SCORE_THRESHOLD = 0.05

data = pd.read_csv("Altered Datasets/ASAP 2.0/Output(Level_0_Only).csv", sep="\t")

num_true_negatives = 0
num_false_positives = 0
# No true positives or false negatives (yet) as the data only includes human-written essays
num_true_positives = 0
num_false_negatives = 0
for i, essay in data.iterrows():
    score, bucket = essay['score_original'], essay['bucket_original']
    # predicted_ai = score > SCORE_THRESHOLD
    predicted_ai = bucket > 0
    is_ai = False # TODO: Replace with according condition once ai-generated text is added
    if is_ai:
        if predicted_ai:
            num_true_positives += 1
        else:
            num_false_negatives += 1
    else:
        if predicted_ai:
            num_false_positives += 1
        else:
            num_true_negatives += 1

num_predicted_ai = num_true_positives + num_false_positives
num_is_ai = num_true_positives + num_false_negatives
num_predicted_correct = num_true_positives + num_true_negatives
num_total = num_true_positives + num_true_negatives + num_false_positives + num_false_negatives

accuracy = num_predicted_correct / num_total
precision = 0.0 if num_predicted_ai == 0 else num_true_positives / num_predicted_ai
recall = 0.0 if num_is_ai == 0 else num_true_positives / num_is_ai
f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall / (precision + recall))
print(f'''Number of true positives (is ai and predicted ai): {num_true_positives}
Number of false negatives (is ai and predicted human): {num_false_negatives}
Number of true negatives (is human and predicted human): {num_true_negatives}
Number of false positives (is human and predicted ai): {num_false_positives}
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1: {f1}''')
