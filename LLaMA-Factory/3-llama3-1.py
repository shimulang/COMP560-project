# sys.path.append('codes/Matterport3DSimulator/build/')
import json
import re

import jsonlines
import requests
from transformers import BertTokenizer, BertModel
import math
import ast
import time
import torch, transformers

model_id = "/root/vln/LLaMA-Factory/saves/llama3-70b/lora_7-14/sft-megred-model/"

# Free up unused memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},  # bfloat16
    device_map="auto"
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# anno_files = ["/root/vln/VLN-DUET/datasets/R2R/annotations/pretrain/train_add_heading_caption_add_elevation.jsonl"]
anno_files = ["/root/vln/VLN-DUET/datasets/R2R/annotations/pretrain/val_seen_caption_add_heading_add_elevation.jsonl"]


for anno_file in anno_files:
    # 创建一个集合来保存新的jsonl文件中的所有索引
    # 保存每个item到新的jsonl文件中
    update_item = []
    with jsonlines.open(anno_file, 'r') as infile:
        for index, item in enumerate(infile):
            print("index: ", index)
            # 如果没有，将这个item添加到新的jsonl文件中
            scan = item["scan"]
            vps = item["path"]
            path_id = item["path_id"]
            captions = item["caption"]
            # generate_instructions = item["generate_instructions"]
            ground_truth_instructions = item["instructions"]
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

            captions_str = str(captions)
            # captions_str = ', '.join(f'"{sentence}"' for sentence in captions)
            PLANNER_PROMPT = (
                f"""   Instructions:
                                    1) Clarity and Self-containment: Each instruction should be self-contained, providing clear directions from one point to the next without using specific angular measurements.
                                    2) Spatial Logic: Follow the spatial logic outlined in the descriptions, aiming to mimic the style and detail orientation of example ground truth instructions provided.
                                    3) Observational Basis: Generate instructions based on the observational descriptions provided, which detail objects and spatial relationships.
                                    4) Instructions Length and Format:  Each Instructions must be a self-contained trajectory.
                       input:{step_descriptions}
                       output：
                      """
            )
            # PLANNER_PROMPT = (
            #     f"""   Instructions: Please generate new instructions based on the  descriptions provided。
            #            input:{step_descriptions}
            #            output：
            #           """
            # )
            # Instruction:
            # time
            start_time = time.time()

            messages = [
                {"role": "user", "content": PLANNER_PROMPT},
            ]

            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            # for i in range(1000):
            outputs = pipeline(
                prompt,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            generate_instructions = outputs[0]["generated_text"][len(prompt):]
            print("Request successful！", generate_instructions)

            print("Ground truth instructions: ", ground_truth_instructions)

            item["generate_instructions"] = generate_instructions

            update_item.append(item)

            print(time.time() - start_time)
    #
    # 保存到新的jsonl文件中
    with jsonlines.open('train_llama3_70B_fineturn_generate_instructions_sub_instruction_718.jsonl', 'w') as f:
        f.write_all(update_item)
