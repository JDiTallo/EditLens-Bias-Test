import pandas as pd
import csv

light = pd.read_csv("Altered Datasets/ASAP 2.0/light_rewrites.tsv", sep="\t")

original = pd.read_csv("Altered Datasets/ASAP 2.0/Output(Level_0_Only).tsv", sep="\t")

# original = original[original["essay_word_count"] <= 1200].reset_index(drop=True)

# score_original	bucket_original	probs_original

light["score_original"] = original["score_original"]
light["bucket_original"] = original["bucket_original"]
light["probs_original"] = original["probs_original"]

desired_order = [
    "essay_id",
    "score",
    "full_text",
    "set",
    "pubpriv",
    "assignment",
    "prompt_name",
    "economically_disadvantaged",
    "student_disability_status",
    "ell_status",
    "race_ethnicity",
    "gender",
    "grade_level",
    "essay_word_count",
    "task",
    "dataset_type",
    "score_original",
    "bucket_original",
    "probs_original",
    "rewritten_essay_light",
    "cosine_distance_light",
    "ngram_score_light",
    "levenshtein_light",
    "score_light",
    "bucket_light",
    "probs_light",
]

light = light[desired_order]

light.to_csv("dataset.tsv", sep="\t", index=False)