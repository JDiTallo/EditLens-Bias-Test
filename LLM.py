import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from editlens_src.scripts.scoring.cosine_distance import cosine_distance
from editlens_src.scripts.scoring.soft_ngrams import soft_ngram_score
from sentence_transformers import SentenceTransformer

import Levenshtein

#Explicitly uses GPU device
device = "cuda"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Loading model and tokenizer...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype="bfloat16",
    attn_implementation="sdpa"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

def rewrite_batch(file_path, 
                  text_feature, 
                  AI_edited_level, 
                  output_path, 
                  batch_size=8, 
                  sep=","):
    
    df = pd.read_csv(file_path, sep=sep)
    df[text_feature] = df[text_feature].fillna("").astype(str)

    all_rewrites = []
    total_rows = len(df)
    
    for i in tqdm(range(0, total_rows, batch_size), desc="Processing Batches", unit="batch"):
        batch_df = df.iloc[i : i + batch_size]
        
        formatted_prompts = []
        for text in batch_df[text_feature]:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": f"Rewrite the following essay:\n\n{text}"}
            ]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(chat_text)

        model_inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        
        trimmed_generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        batch_rewrites = tokenizer.batch_decode(trimmed_generated_ids, skip_special_tokens=True)
        all_rewrites.extend(batch_rewrites)

    df[f"EditLens_rewrite_{AI_edited_level}"] = all_rewrites
    df.to_csv(output_path, sep=sep, index=False)
    print(f"\nDone! Saved to {output_path}")

cosine_model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral", model_kwargs={"dtype": "auto"})
ngram_model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs={"dtype": "auto"})

def evaluate(original_text: str, altered_text: str):
    cosine = cosine_distance(original_text, altered_text)
    ngram = soft_ngram_score(original_text, altered_text, model=ngram_model)
    lev = Levenshtein.distance(original_text, altered_text)

    return cosine, ngram, lev

def add_metrics(file_path, 
                output_path,
                AI_edited_level,
                original_col,
                altered_col,
                batch_size=16, 
                sep=","):
    
    df = pd.read_csv(file_path, sep=sep)
    total_rows = len(df)
    
    all_cosines = []
    all_ngrams = []
    all_levs = []
    
    for i in tqdm(range(0, total_rows, batch_size), desc="Calculating Metrics", unit="batch"):
        batch_df = df.iloc[i : i + batch_size]

        for orig_text, alt_text in zip(batch_df[original_col], batch_df[altered_col]):
            
            #Handle NaNs, just in case
            orig_str = str(orig_text) if pd.notna(orig_text) else ""
            alt_str = str(alt_text) if pd.notna(alt_text) else ""

            cosine, ngram, lev = evaluate(orig_str, alt_str)

            all_cosines.append(cosine)
            all_ngrams.append(ngram)
            all_levs.append(lev)


    df[f'EditLens_level_{AI_edited_level}_cosine'] = all_cosines
    df[f'EditLens_bucket_{AI_edited_level}_ngram'] = all_ngrams
    df[f'EditLens_probs_{AI_edited_level}_levenshtein'] = all_levs

    df.to_csv(output_path, sep=sep, index=False)
    print(f"\nMetrics calculation complete! Saved to {output_path}")

if __name__ == "__main__":

    rewrite_batch(
        file_path="datasets/ASAP_2.0/combined_ASAP_2.csv", 
        output_path="test_rewritten.csv", #Intermediate file
        AI_edited_level=1,
        text_feature="full_text",
        batch_size=16,
        sep="\t"
    )
    
    add_metrics(
        file_path="test_rewritten.csv",
        output_path="test_final.csv",
        AI_edited_level=1,
        original_col="full_text",
        altered_col="EditLens_rewrite_1",
        batch_size=16,
        sep="\t"
    )