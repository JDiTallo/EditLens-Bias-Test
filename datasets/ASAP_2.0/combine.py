import pandas as pd
import csv

# Check if the file is actually tab-separated
train = pd.read_csv("datasets/ASAP_2.0/ASAP_2_Final_github_train.csv", engine='python', quoting=csv.QUOTE_MINIMAL)
test = pd.read_csv("datasets/ASAP_2.0/ASAP_2_Final_github_test.csv", engine='python', quoting=csv.QUOTE_MINIMAL)

train["dataset_type"] = "train"
test["dataset_type"] = "test"

combined_df = pd.concat([train, test], ignore_index=True)
combined_df.to_csv("datasets/ASAP_2.0/combined_ASAP_2.csv", index=False)