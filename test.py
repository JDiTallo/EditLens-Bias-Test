import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from score import score_text, model, tokenizer, n_buckets, DEVICE, MAX_LENGTH, clean_text
from scipy.special import softmax

def test(file_path: str, 
         text_feature: str, 
         AI_edited_level, 
         output_path: str, 
         sep="\t"):
    
    df = pd.read_csv(file_path, sep=sep)
    df[text_feature] = df[text_feature].fillna("").astype(str)

    BATCH_SIZE = 64  
    scores_list = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Scoring dataset"):
        batch_texts = df[text_feature].iloc[i : i + BATCH_SIZE].tolist()
        
        cleaned_texts = [clean_text(text) for text in batch_texts]
        
        inputs = tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,          
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = model(**inputs).logits.float().cpu().numpy()
            
        probs_batch = softmax(logits, axis=1)
        
        for probs in probs_batch:
            bucket = int(probs.argmax())
            score = float((probs * np.arange(n_buckets)).sum() / (n_buckets - 1))
            
            scores_list.append({
                f'score_{AI_edited_level}': score,
                f'bucket_{AI_edited_level}': bucket,
                f'probs_{AI_edited_level}': probs.tolist()
            })

    scores_df = pd.DataFrame(scores_list)
    df[f'score_{AI_edited_level}'] = scores_df[f'score_{AI_edited_level}'].values
    df[f'bucket_{AI_edited_level}'] = scores_df[f'bucket_{AI_edited_level}'].values
    df[f'probs_{AI_edited_level}'] = scores_df[f'probs_{AI_edited_level}'].values

    df.to_csv(output_path, sep=sep, index=False)
