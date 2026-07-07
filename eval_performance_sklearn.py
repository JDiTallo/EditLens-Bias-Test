import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import statistics

SCORE_THRESHOLD = 0.05

data = pd.read_csv("Altered Datasets/ASAP 2.0/Output(Level_0_Only).csv", sep="\t")

true = [False] * len(data)
predicted = [essay['EditLens_bucket_0'] > 0 for _, essay in data.iterrows()]

confusion = confusion_matrix(true, predicted)
num_true_negatives = confusion[0][0]
num_false_positives = confusion[0][1]
num_false_negatives = confusion[1][0]
num_true_positives = confusion[1][1]

accuracy = accuracy_score(true, predicted)
precision = precision_score(true, predicted)
recall = recall_score(true, predicted)
f1 = f1_score(true, predicted)

print(f'''Number of true positives (is ai and predicted ai): {num_true_positives}
Number of false negatives (is ai and predicted human): {num_false_negatives}
Number of true negatives (is human and predicted human): {num_true_negatives}
Number of false positives (is human and predicted ai): {num_false_positives}
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1: {f1}''')
