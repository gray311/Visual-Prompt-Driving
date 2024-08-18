import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return

    filtered_anns = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        if y <= 400 or y > 1500 or x < 200 or x > 1400 or ann['area'] < 500: continue
        filtered_anns.append(ann)

    sorted_anns = sorted(filtered_anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        bbox = ann['bbox']
        color_mask = np.concatenate([np.random.random(3), [0.5]])

        x, y, width, height = bbox
        center_x = x + width // 2
        center_y = y + height // 2

        font_scale, thickness = 0.5, 2
        text = str(idx)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = int(center_x - text_size[0] // 2)
        text_y = int(center_y + text_size[1] // 2)

        img[m] = color_mask 

        cv2.rectangle(img, (text_x - 3, text_y + 3), 
                      (text_x + text_size[0] + 3, text_y - text_size[1] - 3), 
                      (0, 0, 0, 1), -1)
        
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (color_mask[0], color_mask[1], color_mask[2], 1), thickness)
        
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)



if __name__ == "__main__":
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=16,
        points_per_batch=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=50,
        use_m2m=True,
    )

    root = "./images"

    for image_name in os.listdir(root):
        if "jpg" not in image_name: continue
        if "n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448750912460" not in image_name: continue
        image_path = os.path.join(root, image_name)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        masks = mask_generator.generate(image)

        plt.figure(figsize=(40, 40))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(f"./outputs/{image_name}", bbox_inches='tight' , pad_inches=0.0)

        