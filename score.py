import os
import sys

import numpy as np
import torch
from huggingface_hub import login
from peft import PeftModel
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16  # Native fast precision for enterprise GPUs
else:
    try:
        import torch_directml
        DEVICE = torch_directml.device()
        DTYPE = torch.float32  # DirectML functions best at standard float32
    except ImportError:
        # Fallback default for local Windows PC without extra setup
        DEVICE = "cpu"
        DTYPE = torch.float32  # CPUs require float32 to prevent runtime crashes

# Folder containing the EditLens files used
sys.path.insert(0, "editlens_src")
sys.path.insert(0, "editlens_src/scripts")

from preprocess import clean_text          # Normalize text for processing
from train import NormedLinear             # Classification
from inference import infer_n_buckets      # Number of buckets the model outputs to

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER    = "pangram/editlens_Llama-3.2-3B"
MAX_LENGTH = 1024

login(token=HF_TOKEN)

def load_model():

    # How many input catagories (buckets) exists
    n_buckets = infer_n_buckets(ADAPTER)
    # print(f"There are {n_buckets} classification buckets in the adapter")

    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Loads the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=n_buckets,
        torch_dtype=DTYPE,
    )
    base_model = base_model.to(DEVICE)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Replace Linear head with NormedLinear (from train.py)
    if hasattr(base_model, "score") and isinstance(base_model.score, torch.nn.Linear):
        hidden_size = base_model.config.hidden_size
        base_model.score = NormedLinear(hidden_size, n_buckets, device=DEVICE, dtype=DTYPE)

    # Overlays the adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, ADAPTER)
    model.eval()
    return model, tokenizer, n_buckets


model, tokenizer, n_buckets = load_model()

@torch.no_grad()  # No gradients needed, as the model is just scoring
def score_text(text: str) -> dict:

    """
    The model should return three things
      score: from 0 to 1 (or entirely human to entirely AI)
      bucket: the bucket the model believes the text belongs to
      probs: the probabilities for each bucket
    """

    #Normalize the text
    inputs = tokenizer(
        clean_text(text),
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    #Runs the model
    logits = model(**inputs).logits.float().cpu().numpy()

    # Softmax converts logits into probabilities that sum to 1.0
    probs  = softmax(logits, axis=1)[0]

    #The most likely bucket the text belongs in
    bucket = int(probs.argmax())

    score  = float((probs * np.arange(n_buckets)).sum() / (n_buckets - 1))

    return {"score": score, "bucket": bucket, "probs": probs.tolist()}
