# sys.path.append('codes/Matterport3DSimulator/build/')
import json
import re

import jsonlines
import requests
from transformers import BertTokenizer, BertModel
import math

def radian_to_direction(heading_radian, elevation_radian):
    heading_degree = math.degrees(heading_radian)
    elevation_degree = math.degrees(elevation_radian)

    if heading_degree < 0:
        heading_degree += 360


    if 0 <= heading_degree < 90:
        direction = 'turn {} degrees right'.format(heading_degree)
    elif 90 <= heading_degree < 180:
        direction = 'turn {} degrees back'.format(heading_degree - 90)
    elif 180 <= heading_degree < 270:
        direction = 'turn {} degrees left'.format(heading_degree - 180)
    else:
        direction = 'move forward {} degrees'.format(heading_degree - 270)

    if elevation_degree > 0:
        direction += ' and look up {} degrees'.format(elevation_degree)
    elif elevation_degree < 0:
        direction += ' and look down {} degrees'.format(abs(elevation_degree))
    else:
        direction += ' without moving'

    return direction
S = "elevations"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# anno_files = ["datasets/R2R/annotations/pretrain/train_add_heading_caption_add_elevation.jsonl"]
anno_files = ["datasets/R2R/annotations/pretrain/val_seen_caption_add_heading_add_elevation.jsonl"]
url = "https://api.baichuan-ai.com/v1/chat/completions"
api_key = "f0ae37d1e05067c9a086a231f234ad8c"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + api_key
}

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
                    description += f"To your {abs(head)} degrees left"

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
            print("captions_str:{}".format(captions_str))

            # step_descriptions = [
            #     f"The current heading angle is {head} degree,with an elevation of {elev} degree  is {desc}."
            #     for head, elev , desc, in zip(heading_direction_list, elevation_directions, captions)
            # ]
            # examples_captions_str = "\n- " + "\n- ".join(examples_caption)
            # examples_ground_truth_str = "\n- \"" + "\"\n- \"".join(examples_instructions) + "\""
            # 构造完整的prompt
            PLANNER_PROMPT = (
                f"""
                    You are an agent tasked with navigating an indoor environment according to a detailed action plan.
                
                    You should:
                    1） Each instruction should be self-contained, providing clear directions from one point to the next. 
                    2） Present each instruction within quotation marks to indicate its beginning and end clearly. 
                    3) Follow the spatial logic outlined in the descriptions and aim to mimic the style and detail orientation of the example ground truth instructions provided.
                    4） Instructions should be generated based on the Observations descriptions ， headings and elevations provided.
                    5) The instructions you need should contain 3 to 5 sentences, with a total word count of less than 75 words.
                   
                    Here's an example:
                    Steps：['Moving forward, with horizontal is a door with a picture frame and a picture hanging on the wall.', 
                    'To your 270.0 degrees left, with horizontal is a painting hanging on a wall next to a doorway.', 
                    'To your 30.0 degrees right, with horizontal is a close up of a door handle on a white door.', 
                    'To your 60.0 degrees right, with horizontal is a living room with a white couch and a large window.', 
                    'Moving forward, with horizontal is a living room with a couch a chair and a window.', 
                    'To your 30.0 degrees right, with horizontal is a modern living room with a view of the city.']
                    
                    Thought：['walk through the walkway to the left of the wired man, and take your first right into a room. walk into the room, and stop once you reach the gray carpet lying on the ground.',
                    'go into the hallway and to the right. wait in the office, in front of the loveseat.',
                    'exit the room to the left of the statue. turn right into the first door on the right. go straight and walk into the sitting area and stop near the rug on the floor.']
                    Please generate new instructions based on the  descriptions provided。
                    Begin!
                    
                    
                    Steps:{step_descriptions}
                    Thought:["",""]
                   """
                   )
            # Instruction:


            data = {
                "model": "Baichuan2-Turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": PLANNER_PROMPT
                    }
                ],
                "stream": False
            }

            json_data = json.dumps(data)

            response = requests.post(url, data=json_data, headers=headers, timeout=60)

            if response.status_code == 200:
                print("Request successful！")

                result = json.loads(response.text)
                # print("响应结果:", result["choices"][0]["message"]["content"])
                response_content = [result["choices"][0]["message"]["content"]]

                cleaned_sentences = response_content[0].split('\n')
                # 移除空字符串
                list_of_sentences = [sentence.strip() for sentence in cleaned_sentences if sentence]
                # 使用正则表达式匹配并删除数字编号
                # 使用正则表达式去除句子中的编号
                list_cleaned_sentences = [re.sub(r'\d+\. ', '', s) for s in list_of_sentences]
                # del list_cleaned_sentences[0]

                # 创建一个新的列表来存储标记化后的文本
                tokenized_text_list = []

                for i, text in enumerate(list_cleaned_sentences):
                    tokenized_text = tokenizer(text, return_tensors='pt')
                    # 将标记化后的文本添加到新的列表中
                    tokenized_text_list.append(tokenized_text["input_ids"].cpu().detach().numpy().tolist())

                # item["generate_instructions"] = list_cleaned_sentences
                # print("generate_instructions:", list_cleaned_sentences)
                # item["generate_instr_encodings"] = tokenized_text_list

                update_generate_instructions = []
                update_generate_instr_encodings = []
                for index, (generate_instructioning, generate_instr_encoding) in enumerate(
                        zip(list_cleaned_sentences, tokenized_text_list)):
                    # if len(generate_instr_encoding[0]) < 10:
                    #     print(generate_instr_encoding[0])
                    #     continue
                    update_generate_instructions.append(generate_instructioning.strip("\""))
                    tokenized_text = tokenizer(generate_instructioning.strip("\""), return_tensors='pt')
                    update_generate_instr_encodings.append(
                        tokenized_text["input_ids"].cpu().detach().numpy().tolist()[0])

                item["generate_instructions"] = update_generate_instructions
                item["generate_instr_encodings"] = update_generate_instr_encodings



                # cleaned_content = re.sub(r'\b\d+\.\d+\.\d+\b', '', response_content)
                print("successful，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
            else:
                print("Request failed，status_code:", response.status_code)

            update_item.append(item)

    # 保存到新的jsonl文件中
    with jsonlines.open(anno_file.replace('.jsonl', '_baichuan_generate_instructions_v5.jsonl'), 'w') as f:
        f.write_all(update_item)
