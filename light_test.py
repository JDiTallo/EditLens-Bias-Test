from score import score_text
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def test(file_path, text_feature, AI_edited_level, output_path):
  df = pd.read_csv(file_path)
  df[text_feature] = df[text_feature].fillna("").astype(str)

  scores_list = df[text_feature].progress_apply(score_text).tolist()
  scores = pd.DataFrame(scores_list)

  df[f"score_{AI_edited_level}"] = scores['score']
  df[f"bucket_{AI_edited_level}"] = scores['bucket']
  df[f"probs_{AI_edited_level}"] = scores['probs']

  df.to_csv(output_path, sep="\t", index=False)
