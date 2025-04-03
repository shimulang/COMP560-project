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

model_id = "/root/vln/LLaMA-Factory/saves/llama3-70b/lora/sft-megred-model/"

# Free up unused memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16}, #  bfloat16
    device_map="auto"
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

anno_files = ["/root/vln/VLN-DUET/datasets/R2R/annotations/pretrain/train_add_heading_caption_add_elevation.jsonl"]
# anno_files = ["datasets/R2R/annotations/pretrain/val_seen_caption_add_heading_add_elevation.jsonl"]


for anno_file in anno_files:
    # 创建一个集合来保存新的jsonl文件中的所有索引
    indices = set()
    # 打开新的jsonl文件，准备读取数据
    with jsonlines.open('train_llama3_70B_fineturn_generate_instructions.jsonl', 'r') as infile:
        # 遍历新的jsonl文件中的每一行
        for item in infile:
            # 将这个索引添加到集合中
            indices.add(item['path_id'])

    with jsonlines.open('train_llama3_70B_fineturn_generate_instructions.jsonl', 'a') as outfile:
        # 保存每个item到新的jsonl文件中
        update_item = []
        with jsonlines.open(anno_file, 'r') as infile:
            for index, item in enumerate(infile):
                # if index > 10:
                #     break

                print("index: ", index)
                # 检查这个索引是否已经存在于新的jsonl文件中
                if item['path_id'] not in indices:
                    # 如果没有，将这个item添加到新的jsonl文件中
                    scan = item["scan"]
                    vps = item["path"]
                    path_id = item["path_id"]
                    captions = item["caption"]
                    # generate_instructions = item["generate_instructions"]
                    ground_truth_instructions = item["instructions"]
                    # generate_instr_encodings = item["generate_instr_encodings"]
                    heading_degree_list = item["steering_headings"]
                    elevation_degree_list = item["elevations"]
                    # 将heading和elevation弧度转换为角度，如果小于0表示turen left * degress，和captions list 一一对应，作为输入
                    # 将弧度转换为角度
                    # degree_list = [math.degrees(radian) for radian in headings]

                    # 根据角度的大小决定是向右转多少度
                    # direction_list = [f'turn right {degree} degrees' if degree > 0 else f'turn left {degree} degrees' for degree in heading_degree_list]

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


                    PLANNER_PROMPT = (
                        f"""
                               Task Description: You are an agent tasked with navigating an indoor environment according to a detailed action plan.
        
                               Instructions::
                                   1) Clarity and Self-containment: Each instruction should be self-contained, providing clear directions from one point to the next without using specific angular measurements.
                                   2) Quotation Marks: Present each instruction within quotation marks to clearly indicate its beginning and end.
                                   3) Spatial Logic: Follow the spatial logic outlined in the descriptions, aiming to mimic the style and detail orientation of example ground truth instructions provided.
                                   4) Observational Basis: Generate instructions based on the observational descriptions provided, which detail objects and spatial relationships.
                                   5) Instructions Length and Format:  Each Instructions must be a self-contained trajectory containing 3 to 5 sentences, and the total word count of all Instructions must be less than 50 words.
                                   don’t repeat environment that has been described in previous sentence.
        
                               Example Steps Formatting::
                               Steps：['Moving forward, with horizontal is a door with a picture frame and a picture hanging on the wall.', 
                                   'To your 270.0 degrees left, with horizontal is a painting hanging on a wall next to a doorway.', 
                                   'To your 30.0 degrees right, with horizontal is a close up of a door handle on a white door.', 
                                   'To your 60.0 degrees right, with horizontal is a living room with a white couch and a large window.', 
                                   'Moving forward, with horizontal is a living room with a couch a chair and a window.', 
                                   'To your 30.0 degrees right, with horizontal is a modern living room with a view of the city.']
        
                               Example  instructions：walk through the walkway to the left of the wired man, and take your first right into a room. walk into the room, and stop once you reach the gray carpet lying on the ground.
        
                               Please generate new instructions based on the  descriptions provided。
        
        
                               Begin!
                               Generating New Instructions:
                               Steps:{step_descriptions}
                               instruction：[""，"",...]
                              """
                    )
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


                    item["generate_instructions"] = generate_instructions

                    outfile.write(item)

                    print(time.time() - start_time)
    #
    # # 保存到新的jsonl文件中
    # with jsonlines.open(anno_file.replace('.jsonl', '_llama3_8B_generate_instructions_v7.jsonl'), 'w') as f:
    #     f.write_all(update_item)
