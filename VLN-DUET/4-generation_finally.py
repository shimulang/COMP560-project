import json
import re

import jsonlines
import requests
from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# anno_files = ["datasets/R2R/annotations/pretrain/train_llama3_70b_generate_instructionsv7.jsonl"]
anno_files = ["datasets/R2R/annotations/FGR2R_pretrain/train_llama3_70B_fineturn_generate_instructions_sub_instruction_clean_718.jsonl"]

for anno_file in anno_files:
    # 保存每个item到新的jsonl文件中
    update_item = []
    with jsonlines.open(anno_file, 'r') as f:
        for index, item in enumerate(f):
            # if index > 10:
            #     break
            print("index: ", index)
            scan = item["scan"]
            vps = item["path"]
            path_id = item["path_id"]
            print(path_id)
            captions = item["caption"]
            # ground_truth_instructions = item["instructions"]
            generate_instructions = item["generate_instructions"]
            # ground_truth_instructions = item["instructions"]
            generate_instr_encodings = item["generate_instr_encodings"]

            for index,(generate_instructioning, generate_instr_encoding) in enumerate(zip(generate_instructions, generate_instr_encodings)):
                # if len(generate_instr_encoding)<10:
                #     print(generate_instr_encoding)
                #     continue

                new_item = {}
                new_item['path_id'] = path_id
                new_item['instr_id'] = path_id + "_" + str(index)
                new_item['scan'] = scan
                new_item['path'] = vps
                new_item['heading'] = item["heading"]

                new_item["instructions"] = [generate_instructioning]
                new_item["instr_encoding"] = generate_instr_encoding
                update_item.append(new_item)

            print()

        # 保存到新的jsonl文件中
    # with jsonlines.open("datasets/R2R/annotations/pretrain/train_generate_instructions_llama3_70b_finally_v7_6-6.jsonl", 'w') as f:
    #     f.write_all(update_item)
    with jsonlines.open("datasets/R2R/annotations/FGR2R_pretrain/train_llama3_70B_fineturn_generate_instructions_sub_instruction_clean_finally_718.jsonl", 'w') as f:
        f.write_all(update_item)