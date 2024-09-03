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


def laneline_segment(model, image_path, mode="yolo", threshold=200, plot_laneline=True):
    results = model(image_path, conf=0.3)

    for result in results:
        boxes = result.boxes  
        masks = result.masks 
    
    if masks is None:
        return None, None, None
 
    width, height = masks.shape[1:]
    laneline_xy = sorted(masks.xy, key=(lambda x: np.sum(x, axis=0)[0] / x.shape[0]), reverse=False)
    laneline_xy = [np.array([[0.0, height/2.0]])] + laneline_xy + [np.array([[width, height/2.0]])]
    
    laneline_masks = F.interpolate(masks.data.unsqueeze(1), size=masks.orig_shape, mode='nearest').squeeze(1)

    return laneline_masks