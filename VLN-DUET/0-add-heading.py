import json
import jsonlines

# 加载第一个JSONL文件
first_anno_file = "datasets/R2R/annotations/pretrain/train_prevalent_generated_caption.jsonl"
update_item = []
with jsonlines.open(first_anno_file, 'r') as f:
    # 加载第二个JSON数据结构
    with jsonlines.open('datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl', 'r') as file:
        second_data = file

        for index, entry in enumerate(f):
            # 获取path_id
            path_id = entry['path_id']

            # 在第二个JSON数据结构中查找具有相同path_id的条目
            for index1, item in enumerate(second_data):
                # if int(item["instr_id"]) == int(path_id):
                if path_id + '_0' in item['instr_id']:
                    # 找到了匹配的条目，提取heading
                    heading = item['heading']
                    entry['heading'] = heading  # 添加heading到第一个JSONL文件的条目中
                    # entry['instr_id'] = entry['path_id'] # 添加instr_id到第一个JSONL文件的条目中
                    update_item.insert(3, entry)
                    # update_item.append(entry)
                    break
            else:
                # 如果没有找到匹配的条目，打印一条消息
                print(f"No match found for path_id {path_id}")



# 保存到新的jsonl文件中
with jsonlines.open(first_anno_file.replace('.jsonl', '_add_heading.jsonl'), 'w') as f:
    f.write_all(update_item)