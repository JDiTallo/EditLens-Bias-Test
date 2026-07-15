from test import test
from LLM import rewrite_batch, add_metrics
from combine import combine

rewrite_batch(file_path="Altered Datasets/ASAP 2.0/dataset.tsv",
               text_feature="full_text", 
               AI_edited_level="light", 
               output_path="dataset_rewritten.tsv", 
               batch_size=16,
               input_sep="\t",
               output_sep="\t",
               prompt="Correct this essay's grammar, spelling, and punctuation, but retain a similar word choice, sentence structure, content, etc. Also, only output the contents of the essay, and nothing more:\n\n")

add_metrics(
        file_path="dataset_rewritten.tsv",
        output_path="dataset_again.tsv",
        AI_edited_level="light",
        original_col="full_text",
        altered_col="rewritten_essay_light",
        batch_size=16,
        input_sep="\t",
        output_sep="\t"
    )

test(file_path="dataset_again.tsv", 
     output_path="light_final.tsv", 
     AI_edited_level="light",
     text_feature="rewritten_essay_light",
     sep="\t")