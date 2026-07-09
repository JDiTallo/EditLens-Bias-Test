import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Explicitly uses GPU device
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

def rewrite_batch(file_path, text_feature, AI_edited_level, output_path, batch_size=8):
    df = pd.read_csv(file_path).head(5)
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
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nDone! Saved to {output_path}")


if __name__ == "__main__":
    rewrite_batch(
        file_path="datasets/ASAP_2.0/combined_ASAP_2.csv", 
        output_path="test.csv", 
        AI_edited_level=0,
        text_feature="full_text",
        batch_size=16
    )