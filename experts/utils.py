import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import supervision as sv
from collections import defaultdict
import copy
import json
import pickle
from PIL import Image


def concat_image(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, height1))

    return new_image

def has_image_files(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        if file.lower().endswith('.jpg'):
            return True
    
    return False

def calculate_iou(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)

    inter_area = inter_width * inter_height

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    min_area = min(area_box1, area_box2)
    overlap_ratio = inter_area / float(min_area) if min_area > 0 else 0

    return overlap_ratio, area_box1, area_box2


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


def get_filename(path):
    frame_names = [
        p for p in os.listdir(path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return frame_names


def bounding_boxes_close(box1, box2, threshold=50):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    left = max(x_min1, x_min2)
    right = min(x_max1, x_max2)
    top = max(y_min1, y_min2)
    bottom = min(y_max1, y_max2)

    if left < right and top < bottom:
        return True
    
    distance = min(abs(x_min1 - x_max2), abs(x_max1 - x_min2), abs(y_min1 - y_max2), abs(y_max1 - y_min2))
    return distance <= threshold

def bounding_box_to_mask(image_shape, bounding_boxes):
    masks = []
    for box in bounding_boxes:
        mask = np.zeros(image_shape, dtype=bool)
        x_min, y_min, x_max, y_max = box
        mask[int(x_min):int(x_max), int(y_min):int(y_max)] = True
        masks.append(np.array(mask).T)
    return masks