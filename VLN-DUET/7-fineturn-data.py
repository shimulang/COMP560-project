import json
import re

import jsonlines
import requests
from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

anno_files = ["datasets/R2R/annotations/pretrain/train_openai_chatgpt4_generate_instructionsv5.jsonl"]

for anno_file in anno_files:
    # 保存每个item到新的jsonl文件中
    update_item = []
    results = []
    instruction = '''1) Clarity and Self-containment: Each instruction should be self-contained, providing clear directions from one point to the next without using specific angular measurements.
                  2) Quotation Marks: Present each instruction within quotation marks to clearly indicate its beginning and end.
                  3) Spatial Logic: Follow the spatial logic outlined in the descriptions, aiming to mimic the style and detail orientation of example ground truth instructions provided.
                  4) Observational Basis: Generate instructions based on the observational descriptions provided, which detail objects and spatial relationships.
                  5) Instructions Length and Format:  Each Instructions must be a self-contained trajectory containing 3 to 5 sentences, and the total word count of all Instructions must be less than 50 words.don’t repeat environment that has been described in previous sentence.'''
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
            # generate_instructions = item["generate_instructions"]
            ground_truth_instructions = item["instructions"]
            # generate_instr_encodings = item["generate_instr_encodings"]
            heading_degree_list = item["steering_headings"]
            elevation_degree_list = item["elevations"]

            step_descriptions = []
            # [-9.202333567250173, 0.0, 29.999999999999964, 30.000000000000018, -330.0]
            for head, elev, desc in zip(heading_degree_list, elevation_degree_list, captions):
                description = ""
                if -15 < head < 15:
                    description += f"Moving forward"
                elif 165 <= head <= 195 or -195 <= head <= -165:
                    description += f"Moving backward"
                elif 15 <= head < 165 or -165 < head <= -15:
                    description += f"To your {abs(head)} degrees right"
                elif 195 < head < 345 or -345 < head < -195:
                    description += f"To your {360 - abs(head)} degrees left"

                if elev > 0:
                    description += f', with {elev} degrees up'
                elif elev < 0:
                    description += f', with {abs(elev)} degrees down'
                else:
                    description += f', with horizontal'
                description += f' is {desc}.'
                step_descriptions.append(description)


            # 将句子拼接成一个字符串，每个句子用双引号括起来并用逗号分隔
            steps_description = ', '.join(f'"{sentence}"' for sentence in step_descriptions)
            generate_ins = ', '.join(f'"{sentence}"' for sentence in generate_instructions)
            ground_truth_ins = ', '.join(f'"{sentence}"' for sentence in ground_truth_instructions)

            results.append({
                "instruction": instruction,
                "input": steps_description,
                "output": generate_ins
            })
            # results.append({
            #     "instruction": instruction,
            #     "input": steps_description,
            #     "output": ground_truth_ins
            # })

            print()

    # 将列表保存为 JSON 文件
    file_path = 'datasets/R2R/annotations/pretrain/r2r_vln_fineturn_data.json'
    with open(file_path, 'w') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    # 输出保存的 JSON 文件路径
    print(f"The sentences have been saved to {file_path}")