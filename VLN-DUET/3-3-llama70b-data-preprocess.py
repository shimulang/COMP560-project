import json
import re

import jsonlines
import requests
from transformers import BertTokenizer, BertModel
import ast

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

anno_files = ["datasets/R2R/annotations/pretrain/train_prevalent_generated_caption_add_heading_add_elevation_llama3_70B_generate_instructions_v7.jsonl"]

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
            # ground_truth_instructions = item["instructions"]
            # generate_instr_encodings = item["generate_instr_encodings"]

            # 使用正则表达式找到所有的指令
            pattern = r'\"(.*?)\"'
            cleaned_instructions = re.findall(pattern, generate_instructions)

            # # 找到第一个 [ 的位置
            # start = generate_instructions.find('[')
            # # 找到最后一个 ] 的位置
            # end = generate_instructions.rfind(']') + 1
            #
            # # 提取出 [] 中的内容
            # cleaned_sentences = generate_instructions[start:end]
            #
            # cleaned_instructions= []
            # if start == -1 or end == 0:
            #     # instructions = re.findall(r'\d+\) "[^"]+"', generate_instructions)
            #     instructions = re.findall(r'"\s*([^"]+)\s*"', generate_instructions)
            #     cleaned_instructions = [instr.strip(")").strip().strip('"') for instr in instructions]
            # else:
            #     # 使用 ast.literal_eval 将字符串转化为 list
            #     cleaned_instructions = ast.literal_eval(cleaned_sentences)

            # # 使用 ast.literal_eval 将字符串转化为 list
            # cleaned_instructions = ast.literal_eval(cleaned_sentences)
            print("generate instructions:", cleaned_instructions)
            generate_instr_encodings = []
            for index, gen_instru in enumerate(cleaned_instructions):
                tokenized_text = tokenizer(gen_instru, return_tensors='pt')
                generate_instr_encodings.append(tokenized_text["input_ids"].cpu().detach().numpy().tolist()[0])


            # update_generate_instructions = []
            # update_generate_instr_encodings = []
            # for index,(generate_instructioning, generate_instr_encoding) in enumerate(zip(generate_instructions, generate_instr_encodings)):
            #     if len(generate_instr_encoding[0])<10:
            #         print(generate_instr_encoding[0])
            #         continue
            #     update_generate_instructions.append(generate_instructioning.strip("\""))
            #     tokenized_text = tokenizer(generate_instructioning.strip("\""), return_tensors='pt')
            #     update_generate_instr_encodings.append(tokenized_text["input_ids"].cpu().detach().numpy().tolist()[0])

            item["generate_instructions"] = cleaned_instructions
            item["generate_instr_encodings"] = generate_instr_encodings
            update_item.append(item)
            # for index,(generate_instructioning, generate_instr_encoding) in enumerate(zip(generate_instructions, generate_instr_encodings)):
            #     if len(generate_instr_encoding[0])<10:
            #         print(generate_instr_encoding[0])
            #         continue
            #
            #     new_item = {}
            #     new_item['path_id'] = path_id
            #     new_item['instr_id'] = path_id + "_" + str(index)
            #     new_item['scan'] = scan
            #     new_item['path'] = vps
            #     new_item['heading'] = item["heading"]
            #
            #     new_item["instructions"] = generate_instructioning.strip("\"")
            #     tokenized_text = tokenizer(generate_instructioning.strip("\""), return_tensors='pt')
            #     new_item["instr_encoding"] = tokenized_text["input_ids"].cpu().detach().numpy().tolist()[0]
            #     update_item.append(new_item)

            print()

        # 保存到新的jsonl文件中
    with jsonlines.open("datasets/R2R/annotations/pretrain/train_llama3_70b_generate_instructionsv7.jsonl", 'w') as f:
        f.write_all(update_item)