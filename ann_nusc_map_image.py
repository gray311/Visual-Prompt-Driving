import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
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
        plt.savefig("1.png")

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

import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

import descartes

from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box


map_api = nusc_map = NuScenesMap(dataroot='/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes', map_name='singapore-onenorth')

if __name__ == "__main__":
    sam2_checkpoint = "./workspace/checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)

    predictor = SAM2ImagePredictor(sam2, 
                                   mask_threshold=0.3,
                                    max_hole_area=1000,
                                    max_sprinkle_area=0,
                                   )
    

    # mask_generator = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=16,
    #     points_per_batch=32,
    #     pred_iou_thresh=0.7,
    #     stability_score_thresh=0.92,
    #     stability_score_offset=0.7,
    #     crop_n_layers=1,
    #     box_nms_thresh=0.7,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=50,
    #     use_m2m=True,
    # )

    root = "/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes/samples/CAM_FRONT"
    
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', verbose=False, dataroot='/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes')

    # hyperparameters
    patch_radius = 50
    near_plane = 1e-8
    min_polygon_area: float = 100
    render_outside_im = True

    # Pick a sample and render the front camera image.
    sample_token = nusc.sample[3]['token']
    # layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    layer_names = ['lane']
    camera_channel = 'CAM_FRONT'
    # nusc_map.render_map_in_image(nusc, sample_token, layer_names=layer_names, camera_channel=camera_channel)


    sample_record = nusc.get('sample', sample_token)
    scene_record = nusc.get('scene', sample_record['scene_token'])
    log_record = nusc.get('log', scene_record['log_token'])
    log_location = log_record['location']

    # Grab the front camera image and intrinsics.
    cam_token = sample_record['data'][camera_channel]
    cam_record = nusc.get('sample_data', cam_token)
    cam_path = nusc.get_sample_data_path(cam_token)
    image = im = Image.open(cam_path)
    im_size = im.size
    cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    # Retrieve the current map.
    poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
    ego_pose = poserecord['translation']
    box_coords = (
        ego_pose[0] - patch_radius,
        ego_pose[1] - patch_radius,
        ego_pose[0] + patch_radius,
        ego_pose[1] + patch_radius,
    )
    records_in_patch = map_api.explorer.get_records_in_patch(box_coords, layer_names, 'intersect')
    
    plt.figure(figsize=(40, 40))
    plt.imshow(image)

    for layer_name in layer_names:
        for token in records_in_patch[layer_name]:
            record = map_api.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = record['polygon_tokens']
            else:
                polygon_tokens = [record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = map_api.extract_polygon(polygon_token)

                # Convert polygon nodes to pointcloud with 0 height.
                points = np.array(polygon.exterior.xy)
                points = np.vstack((points, np.zeros((1, points.shape[1]))))

                # Transform into the ego vehicle frame for the timestamp of the image.
                points = points - np.array(poserecord['translation']).reshape((-1, 1))
                points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                # Transform into the camera.
                points = points - np.array(cs_record['translation']).reshape((-1, 1))
                points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

                # Remove points that are partially behind the camera.
                depths = points[2, :]
                behind = depths < near_plane
                if np.all(behind):
                    continue

                # if render_behind_cam:
                    # Perform clipping on polygons that are partially behind the camera.
                points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
                # elif np.any(behind):
                #     # Otherwise ignore any polygon that is partially behind the camera.
                #     continue

                # Ignore polygons with less than 3 points after clipping.
                if len(points) == 0 or points.shape[1] < 3:
                    continue

                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points = view_points(points, cam_intrinsic, normalize=True)

                # Skip polygons where all points are outside the image.
                # Leave a margin of 1 pixel for aesthetic reasons.
                inside = np.ones(points.shape[1], dtype=bool)
                inside = np.logical_and(inside, points[0, :] > 1)
                inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                inside = np.logical_and(inside, points[1, :] > 1)
                inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                if render_outside_im:
                    if np.all(np.logical_not(inside)):
                        continue
                else:
                    if np.any(np.logical_not(inside)):
                        continue

                points = points[:2, :]
                # points[0, :] = np.clip(points[0, :], 0, im.size[0] - 1)
                # points[1, :] = np.clip(points[1, :], 0, im.size[1] - 1)
                points_2d = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                
                polygon_proj = Polygon(points_2d)
                
                polygon_im = Polygon([(0, 0), (0, im.size[1]), (im.size[0], im.size[1]), (im.size[0], 0)])

                # Filter small polygons
                if polygon_proj.area < min_polygon_area:
                    continue

                label = layer_name
                
                # plt.figure(figsize=(40, 40))
                # plt.imshow(image)
                # show_anns(masks)
                # show_masks(image, masks, scores, box_coords=input_box)
                random_color = (random.random(), random.random(), random.random())

                plt.gca().add_patch(descartes.PolygonPatch(polygon_proj, alpha=0.3, color=random_color, label=label))

                if False:
                    predictor.set_image(image)
                    # input_box = np.array([points[0].min(), points[1].min(), points[0].max(), points[1].max()])
                    # input_box = np.array([0,0,1600,800])
                    
                    # input_points = []
                    # input_labels = []
                    # for i in range(points.shape[1]//2):
                    #     # input_points.append([points[0][i+1]+points[0][-i], points[1][i+1]+points[1][-i]])
                    #     input_points.append([points[0][i], points[1][i]])
                    #     input_labels.append(1)
                        
                    # input_points = points[:,:-1].mean(axis=1)
                        
                    input_points = np.array(polygon_proj.intersection(polygon_im).centroid.coords)
                    
                    print("="*50)
                    print(input_points, points)
                    
                    # input_points = np.array([[747,741]])
                    input_labels = np.array([1])
        
                    masks, scores, logits = predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        # box=input_box[None, :],
                        multimask_output=False,
                    )   
                    
                    # sorted_ind = np.argsort(scores)[::-1]
                    # masks = masks[sorted_ind]
                    # scores = scores[sorted_ind]
                    # logits = logits[sorted_ind]
                    
                    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
                    
                    # masks, scores, _ = predictor.predict(
                    #     point_coords=input_points,
                    #     point_labels=input_labels,
                    #     mask_input=mask_input[None, :, :],
                    #     multimask_output=False,
                    # )
                    

                    # masks = mask_generator.generate(image)

                # plt.show()
                # show_masks(image, masks, scores, point_coords=input_points, input_labels=input_labels)
                
                # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
                
                
                # plt.show()
    
    plt.savefig("1.png")
    # for image_name in os.listdir(root):
    #     # print(image_name)
    #     if "jpg" not in image_name: continue
    #     if "n015-2018-10-02-10-50-40+0800__CAM_FRONT__" not in image_name: continue
    #     image_path = os.path.join(root, image_name)
    #     image = Image.open(image_path)
    #     image = np.array(image.convert("RGB"))
        
    #     predictor.set_image(image)
    #     input_box = np.array([225, 300, 700, 875])
    
    #     masks, scores, _ = predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_box[None, :],
    #     multimask_output=False,
    #     )   

    #     # masks = mask_generator.generate(image)

    #     plt.figure(figsize=(40, 40))
    #     plt.imshow(image)
    #     # show_anns(masks)
    #     show_masks(image, masks, scores, box_coords=input_box)
    #     plt.axis('off')
    #     plt.savefig(f"./workspace/map_outputs/{image_name}", bbox_inches='tight' , pad_inches=0.0)
        
    #     break