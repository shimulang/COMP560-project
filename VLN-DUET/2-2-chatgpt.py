# sys.path.append('codes/Matterport3DSimulator/build/')
import json
import re

import jsonlines
import requests
from transformers import BertTokenizer, BertModel
import math
import ast

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

anno_files = ["/root/vln/VLN-DUET/datasets/R2R/annotations/pretrain/train_prevalent_generated_caption_add_heading_add_elevation.jsonl"]
# anno_files = ["datasets/R2R/annotations/pretrain/val_seen_caption_add_heading_add_elevation.jsonl"]
from openai import OpenAI
#？
# client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = "OPENAI_API_KEY"
# )

import openai
# 设置你的OpenAI API密钥
client = OpenAI(
    # This is the default and can be omitted
    api_key="OPENAI_API_KEY",
)



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

            # 根据角度的大小决定是向上还是向下看多少度
            # directions = []
            # for heading_radian, elevation_radian in zip(heading_radian_list, elevations_radian_list):
            #     directions.append(radian_to_direction(heading_radian, elevation_radian))

            # 转换captions为一个格式化的字符串
            # captions_str = "\n- " + "\n- ".join([caption[0] for caption in captions if caption])
            # captions_str = "\n- " + "\n- ".join([caption for caption in captions if caption])
            # captions_str = [caption[0] for caption in captions if caption]
            captions_str = str(captions)
            # print("captions_str:{}".format(captions_str))

            # step_descriptions = [
            #     f"The current heading angle is {head} degree,with an elevation of {elev} degree  is {desc}."
            #     for head, elev , desc, in zip(heading_direction_list, elevation_directions, captions)
            # ]
            # examples_captions_str = "\n- " + "\n- ".join(examples_caption)
            # examples_ground_truth_str = "\n- \"" + "\"\n- \"".join(examples_instructions) + "\""
            # 构造完整的prompt
            # PLANNER_PROMPT = (
            #     f"""
            #         You are an agent tasked with navigating an indoor environment according to a detailed action plan.
            #
            #         You should:
            #             1) Each instruction should be self-contained, providing clear directions from one point to the next without using specific angular measurements.
            #             2) Present each instruction within quotation marks to clearly indicate its beginning and end.
            #             3) Follow the spatial logic outlined in the descriptions, aiming to mimic the style and detail orientation of example ground truth instructions provided.
            #             4) Generate instructions based on the observational descriptions provided, which detail objects and spatial relationships.
            #             5) Each instruction must be a self-contained trajectory containing 3 to 5 sentences, and the total word count of each instruction should not exceed 75 words.
            #
            #
            #         Here's an example:
            #         Steps：['Moving forward, with horizontal is a door with a picture frame and a picture hanging on the wall.',
            #             'To your 270.0 degrees left, with horizontal is a painting hanging on a wall next to a doorway.',
            #             'To your 30.0 degrees right, with horizontal is a close up of a door handle on a white door.',
            #             'To your 60.0 degrees right, with horizontal is a living room with a white couch and a large window.',
            #             'Moving forward, with horizontal is a living room with a couch a chair and a window.',
            #             'To your 30.0 degrees right, with horizontal is a modern living room with a view of the city.']
            #
            #         instruction：['walk through the walkway to the left of the wired man, and take your first right into a room. walk into the room, and stop once you reach the gray carpet lying on the ground.',
            #         'go into the hallway and to the right. wait in the office, in front of the loveseat.',
            #         'exit the room to the left of the statue. turn right into the first door on the right. go straight and walk into the sitting area and stop near the rug on the floor.']
            #         Please generate new instructions based on the  descriptions provided。
            #         Begin!
            #
            #
            #         Steps:{step_descriptions}
            #         instruction：["","",...]
            #        """
            # )

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
                       instruction：
                      """
            )
            # Instruction:

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": PLANNER_PROMPT,
                    }
                ],
                model="gpt-4o",
                # model="gpt-3.5-turbo-0125", gpt-4-turbo
                temperature=0.5,
                    top_p=1,
                    max_tokens=1024,
            )
            # response = client.chat.completions.create(
            #     model="meta/llama3-70b",
            #     messages=[{"role": "user", "content": PLANNER_PROMPT}],
            #     temperature=0.5,
            #     top_p=1,
            #     max_tokens=1024,
            #     # stream=True
            # )

            response_content = response.choices[0].message.content

            generate_instructions = response_content
            print("Request successful！", generate_instructions)

            # 找到第一个 [ 的位置
            # start = response_content.find('[')
            # # 找到最后一个 ] 的位置
            # end = response_content.rfind(']') + 1
            #
            # # 提取出 [] 中的内容
            # cleaned_sentences = response_content[start:end]
            # print("generate instructions:", cleaned_sentences)
            # # 使用 ast.literal_eval 将字符串转化为 list
            # generate_instructions = ast.literal_eval(cleaned_sentences)
            #
            #
            #
            # generate_instr_encodings = []
            # for i, text in enumerate(generate_instructions):
            #     tokenized_text = tokenizer(text, return_tensors='pt')
            #     # 将标记化后的文本添加到新的列表中
            #     generate_instr_encodings.append(
            #         tokenized_text["input_ids"].cpu().detach().numpy().tolist()[0])

            item["generate_instructions"] = generate_instructions
            # item["generate_instr_encodings"] = generate_instr_encodings

            # cleaned_content = re.sub(r'\b\d+\.\d+\.\d+\b', '', response_content)
            # print("successful，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))


            update_item.append(item)

    # 保存到新的jsonl文件中
    with jsonlines.open(anno_file.replace('.jsonl', '_openai_generate_instructions_v6.jsonl'), 'w') as f:
        f.write_all(update_item)
