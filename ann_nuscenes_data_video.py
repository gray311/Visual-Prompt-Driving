import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from dataset.nuscenes_dataset import NuscenesLoader


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__=="__main__":
    sam2_checkpoint = "./workspace/ckpt/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda:1")

    scene_id = 0
    video_path = f"./images/nuscenes_scene_{scene_id}"
    shutil.rmtree(video_path)
    os.mkdir(video_path)

    loader = NuscenesLoader(version="v1.0-mini", dataroot="/home/yingzi/Visual-Prompt-Driving/workspace/nuscenes", frequency=3)
    metadata = loader.load(scene_id)
    print(metadata['scene_description'])

    samples = metadata['sample_descriptions']
    for sample_idx, line in samples.items():
        frame_path = line['filepath']

        image_name = str(sample_idx).zfill(5) + ".jpg"
        image = Image.open(frame_path)
        image.save(os.path.join(video_path, image_name))

    predictor.init_state(video_path=video_path)

    ### generate mask
    ann_frame_idx = 0  # the frame index we interact with







