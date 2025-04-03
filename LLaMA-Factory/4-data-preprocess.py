import json
import re

import jsonlines
import requests
from transformers import BertTokenizer, BertModel
import ast

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

anno_files = ["train_llama3_70B_fineturn_generate_instructions_sub_instruction_718.jsonl"]

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
            captions = item["caption"]
            # ground_truth_instructions = item["instructions"]
            generate_instructions = item["generate_instructions"]

            # 使用 split 方法将字符串按逗号分割成列表，并去除多余的空格和引号
            cleaned_instructions = [s.strip().strip("'").strip('"') for s in generate_instructions.split(', ')]

            print("generate instructions:", cleaned_instructions)
            generate_instr_encodings = []
            for index, gen_instru in enumerate(cleaned_instructions):
                tokenized_text = tokenizer(gen_instru, return_tensors='pt')
                generate_instr_encodings.append(tokenized_text["input_ids"].cpu().detach().numpy().tolist()[0])

            item["generate_instructions"] = cleaned_instructions
            item["generate_instr_encodings"] = generate_instr_encodings
            update_item.append(item)


            print()

        # 保存到新的jsonl文件中
    with jsonlines.open("/root/vln/VLN-DUET/datasets/R2R/annotations/FGR2R_pretrain/train_llama3_70B_fineturn_generate_instructions_sub_instruction_clean_718.jsonl", 'w') as f:
        f.write_all(update_item)