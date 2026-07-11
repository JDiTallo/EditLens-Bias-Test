from light_test import test

test(file_path="datasets\ASAP_2.0\light_rewrites_pared.tsv", 
     output_path="test.csv", 
     AI_edited_level="light",
     text_feature="rewritten_essay_light",
     sep="\t")