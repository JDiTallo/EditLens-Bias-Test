from light_test import test

test(file_path="datasets\ASAP_2.0\combined_ASAP_2.tsv", 
     output_path="test.csv", 
     AI_edited_level="original",
     text_feature="full_text",
     sep="\t")