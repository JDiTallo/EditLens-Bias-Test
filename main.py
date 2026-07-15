from test import test

test(file_path="Altered Datasets/ASAP 2.0/light_rewrites.tsv", 
     output_path="light_rewrites_scored.tsv", 
     AI_edited_level="light",
     text_feature="rewritten_essay_light",
     sep="\t")