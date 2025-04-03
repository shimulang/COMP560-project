
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
# from lavis.models import load_model_and_preprocess
import torch
import json
import jsonlines
import os
from PIL import Image
import shutil

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


# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
# raw_image = Image.open("rgb_batch_0.png").convert("RGB")

# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
# model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
# tmp = model.generate({"image": image})
# print(tmp)
# ['a large fountain spewing water into the air']

NEWHEIGHT = 248
NEWWIDTH = int(WIDTH / HEIGHT * NEWHEIGHT)
print(NEWHEIGHT, NEWWIDTH)

data_size_per_img = np.random.randint(255, size=(NEWHEIGHT, NEWWIDTH, 3), dtype=np.uint8).nbytes

anno_files = ["datasets/R2R/annotations/pretrain/train_prevalent_generated_caption_add_heading.jsonl"]
# anno_files = ["datasets/R2R/annotations/pretrain/val_seen_caption_add_heading.jsonl"]
# save_dir = "../datasets/traj_data"

for anno_file in anno_files:
    # 保存每个item到新的jsonl文件中
    update_item = []
    with jsonlines.open(anno_file, 'r') as f:
        for index, item in enumerate(f):
            print("index: ", index)
            scan = item["scan"]
            vps = item["path"]
            path_id = item["path_id"]
            instructions = item["instructions"]

            path_viewpoints = item["path_viewindex"]
            # item["caption"] = []
            item["current_headings"] = []
            item["elevations"] = []
            item["steering_headings"] = []

            # save_path_id = os.path.join(save_dir, path_id)
            # # 保存每个vp的图片到相应的path_id文件夹中
            # if not os.path.exists(save_path_id):
            #     os.makedirs(save_path_id)

            # 保存每个path_id的instructions和vps到一个txt文件中，每行包括path_id, instructions, vps, path_viewpoints
            init_heading = item["heading"]
            # 上一步的heading
            last_step_heading = init_heading

            for index, (vp,viewindex) in enumerate(zip(vps,path_viewpoints)):
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
                        heading_radian = state.heading
                        heading_degree = math.degrees(heading_radian)
                        item["current_headings"].append(round(heading_degree))

                        # 当前heading与上一步heading的差值作为转向角度
                        steering_heading_radian = heading_radian - last_step_heading
                        steering_heading_degree = math.degrees(steering_heading_radian)
                        item["steering_headings"].append(round(steering_heading_degree))
                        last_step_heading = heading_radian

                        elevation_radian = state.elevation
                        elevation_degree = math.degrees(elevation_radian)
                        item["elevations"].append(round(elevation_degree))
                    # elevation=state.elevation

            # item["elevation"]=elevation
            update_item.append(item)

    # 保存到新的jsonl文件中
    with jsonlines.open(anno_file.replace('.jsonl', '_add_elevation.jsonl'), 'w') as f:
        f.write_all(update_item)
