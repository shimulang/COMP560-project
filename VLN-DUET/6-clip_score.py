import jsonlines

import os
import sys
import numpy as np
import json
import collections
from PIL import Image
import matplotlib.pyplot as plt
import lmdb
import math
import cv2

sys.path.append('/root/vln/Matterport3DSimulator/build/')
import MatterSim
import torch
import jsonlines
from PIL import Image
import open_clip

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60

scan_data_dir = '/root/vln/datasets/v1/mp3d'
connectivity_dir = 'datasets/R2R/connectivity'

sim = MatterSim.Simulator()
sim.setDatasetPath(scan_data_dir)
sim.setNavGraphPath(connectivity_dir)
sim.setPreloadingEnabled(True)
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(math.radians(VFOV))
sim.setDiscretizedViewingAngles(True)
sim.setBatchSize(1)
sim.initialize()

NEWHEIGHT = 248
NEWWIDTH = int(WIDTH / HEIGHT * NEWHEIGHT)
print(NEWHEIGHT, NEWWIDTH)

data_size_per_img = np.random.randint(255, size=(NEWHEIGHT, NEWWIDTH, 3), dtype=np.uint8).nbytes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')



# anno_file = "datasets/R2R/annotations/pretrain/train_llama3_70b_generate_instructionsv7.jsonl"
anno_file = "/root/vln/VLN-DUET/datasets/R2R/annotations/FGR2R_pretrain/train_llama3_70B_fineturn_generate_instructions_sub_instruction_clean_718.jsonl"
all_cosine_similarity = []
num = 0
with jsonlines.open(anno_file, 'r') as f:
    for index, item in enumerate(f):
        scan = item["scan"]
        vps = item["path"]
        path_id = item["path_id"]
        instructions = item["instructions"]
        generate_instructions = item["generate_instructions"]
        # generate_instructions = instructions


        print(scan, path_id)
        path_viewpoints = item["path_viewindex"]
        if len(generate_instructions)==1:
            text = tokenizer(generate_instructions)
        else:
            #将多个句子合并成一个句子
            text_tmp = [" ".join(generate_instructions)]
            text = tokenizer(text_tmp)
        with torch.no_grad(), torch.cuda.amp.autocast():
            # image_features = model.encode_image(image)
            text_features = model.encode_text(text.to(device))

        image_features_list = []
        for index, (vp, viewindex) in enumerate(zip(vps, path_viewpoints)):
            key = '%s_%s' % (scan, vp)
            key_byte = key.encode('ascii')

            for ix in range(36):
                if ix == 0:
                    sim.newEpisode([scan], [vp], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [1.0], [0])
                state = sim.getState()[0]

                if state.viewIndex == viewindex:
                    image = np.array(state.rgb, copy=True)  # in BGR channel
                    image = Image.fromarray(image[:, :, ::-1])  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # resize
                    image = image.resize((NEWWIDTH, NEWHEIGHT), Image.ANTIALIAS)
                    image = np.array(image)

                    raw_image = Image.fromarray(image[:, :, ::-1])  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # 保存图片
                    image = preprocess(raw_image).unsqueeze(0).to(device)

                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_features = model.encode_image(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        image_features_list.append(image_features)
        # 计算视觉特征的平均值
        average_image_features = torch.mean(torch.stack(image_features_list), dim=0)

        # 计算余弦相似度
        cosine_similarity = torch.nn.functional.cosine_similarity(text_features, average_image_features).detach().cpu().numpy()[0]
        all_cosine_similarity.append(cosine_similarity)
        num += 1
# 输出相似度分数0.2732
print("Avg Cosine Similarity Score {}:".format(num), np.mean(all_cosine_similarity))