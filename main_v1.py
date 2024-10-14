import os
import sys
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import supervision as sv
from collections import defaultdict
import cv2
import copy
import json
import pickle
import descartes
import random
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers import AutoProcessor, AutoModelForCausalLM
from dataset.nuscenes_dataset import NuscenesLoader
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils, compute_iou, compute_area, ROAD_OBJECTS, OBJECTS_PRIORITY
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from utils.polygon_dictionary_model import PolygonDictionaryModel, LaneInfo
from ultralytics import YOLO
from experts import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.visualizer import Visualizer, bounding_box_to_mask
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')



def build_model(sam2_checkpoint, model_cfg, grounding_model_name="dino"):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    if grounding_model_name == "dino":
        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    elif grounding_model_name == "florence2":
        FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
        grounding_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True).eval().to(device)
        processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)


    lane_model = YOLO("./workspace/checkpoint/yolov8_seg.pt")

    return image_predictor, video_predictor, grounding_model, lane_model, processor, device


def generate_realtime_data(scene_id, loader, task_name):
    video_path = f"./images/nuscenes_scene_{scene_id}"

    if not os.path.exists(video_path):
        # shutil.rmtree(video_path)
        os.mkdir(video_path)

    # loader = NuscenesLoader(version="v1.0-trainval", dataroot="/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes", frequency=1)
    metadata = loader.load(scene_id)
    samples = metadata['sample_descriptions']
    for sample_idx, line in samples.items():
        frame_path = line['filepath']
        image_name = str(sample_idx).zfill(6) + ".jpg"
        image = Image.open(frame_path)
        image.save(os.path.join(video_path, image_name))
    
    output_dir = f"./outputs/{task_name}/nuscenes_scene_{scene_id}"
    # 'output_video_path' is the path to save the final video
    output_video_path = f"./outputs/{task_name}/nuscenes_scene_{scene_id}/output.mp4"

    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    frame_names = get_filename(video_path)

    return samples, video_path, output_dir, output_video_path, mask_data_dir, json_data_dir, result_dir, frame_names


if __name__=="__main__":
    task_name = "visual_prompt_with_object_lane"
    using_mark = True
    if task_name == "baseline":
        using_mark = False
    scene_id_file = "./dataset/scene_id.json"
    sam2_checkpoint = "./workspace/checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    grounding_model_name = "dino"
    image_predictor, video_predictor, grounding_model, lane_model, processor, device = build_model(sam2_checkpoint, model_cfg, grounding_model_name=grounding_model_name)

    loader = NuscenesLoader(version="v1.0-mini", dataroot="/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes", frequency=1)
    
    with open(scene_id_file, "r") as f:
        scene_id_split = json.load(f)

    overtake_scene_ids = scene_id_split['long_tail']
    # overtake_scene_ids = [1]
    # overtake_scene_ids = [idx for idx in overtake_scene_ids if idx not in [217, 252]]

    print(
        f"There are {len(overtake_scene_ids)} long tail scenes for evaluation."
    )

    from tqdm import tqdm
    for i in tqdm([0, 1, 3]):
        samples, video_dir, output_dir, output_video_path, \
            mask_data_dir, json_data_dir, result_dir, frame_names  = generate_realtime_data(scene_id=i, loader=loader, task_name=task_name)
  
        if not os.path.exists(result_dir):
        # if True:
            # init video predictor state
            inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
            step = 4 # the step to sample frames for Grounding DINO predictor

            sam2_masks = {object_name: MaskDictionaryModel() for object_name in ROAD_OBJECTS}
            objects_count = {object_name: 0 for object_name in ROAD_OBJECTS}
            lane_polygon = PolygonDictionaryModel()
            lane_count = 0
            PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
            use_ground_truth = True

            print(f"Frame num: {len(frame_names)}")
            for start_frame_idx in range(0, len(frame_names), step):
                print("start_frame_idx", start_frame_idx)
                # continue
                img_path = os.path.join(video_dir, frame_names[start_frame_idx])
                image = Image.open(img_path)
                width, height = image.size
                image_base_name = frame_names[start_frame_idx].split(".")[0]
            
                boxes, labels, locations = defaultdict(list), defaultdict(list), defaultdict(list)
                segments = defaultdict(dict)
                sample_token = None
                
                object_max_size = {'pedestrian.': 0, 'truck.': 10000, 'bus.': 10000, 'car.': 1000}
                if use_ground_truth:
                    instances = samples[start_frame_idx]['sample_annotations']
                    remove_id = []
                    for instance_id, item in instances.items():
                        if item['category_name'].split(".")[1].strip(" ") not in ROAD_OBJECTS:
                            remove_id.append(instance_id)
                        elif  "v0-" in item['visibility'] or "v40-" in item['visibility']:
                            remove_id.append(instance_id)
                        elif item['bounding_box'][2] <= 80 or item['bounding_box'][0] >= 1520:
                            remove_id.append(instance_id)

                    object_ids = [instance_id for instance_id, item in instances.items() if instance_id not in remove_id]
                    for instance_id, item in instances.items():
                        if instance_id in remove_id: continue
                        boxes[item['category_name'].split(".")[1].strip(" ")].append(item['bounding_box'])
                        labels[item['category_name'].split(".")[1].strip(" ")].append(item['category_name'].split(".")[1].strip(" "))
                        item['location'].extend(item['size_in_meter'])
                        item['location'].extend([item['instance_token']])
                        locations[item['category_name'].split(".")[1].strip(" ")].append(item['location'])

                    for object_name in ROAD_OBJECTS:
                        if len(boxes[object_name]) == 0:
                            boxes[object_name] = None
                            labels[object_name] = None 
                            locations[object_name] = None 


                # run Grounding DINO on the image
                # we can use driving expert model to detect bounding box or lane segmentation
                elif grounding_model_name == "dino":
                    for object_name in ROAD_OBJECTS:
                        input_boxes, input_labels = dino_detect_object(image, object_name, grounding_model, processor)
                        # Filter out offending bounding boxes
                        tmp_boxes, max_bbox_size = [], 0
                        for bbox in input_boxes:
                            area = compute_area(bbox)
                            if bbox[1] <= 350 and area <= 40000: continue
                            if area <= object_max_size[object_name]: continue
                            if area > max_bbox_size:
                                max_bbox_size = area
                            tmp_boxes.append(bbox)

                        if len(tmp_boxes) == 0:
                            boxes[object_name] = None
                            labels[object_name] = None
                            continue

                        if len(tmp_boxes) <= 2 and max_bbox_size <= 40000:
                            boxes[object_name] = None
                            labels[object_name] = None
                            continue

                        boxes[object_name] = tmp_boxes
                        labels[object_name] = [input_labels[0]] * len(tmp_boxes)
        
                elif grounding_model_name == "florence2":
                    boxes, labels = florence2_detect_object(image, "car <and> truck <and> pedestrain <and> bus", grounding_model, processor)              

                if all(value is None for value in boxes.values()):
                    print(
                        "No new objects are detected in this frame!"
                    )
                    continue
                # Remove the occluded objects.
                for obj_name1 in ROAD_OBJECTS:
                    if use_ground_truth == True: continue # If ground truth is used, no filtering is required.
                    bbox1 = boxes[obj_name1]
                    if bbox1 is None: continue
                    tmp_boxes = []
                    for i in range(len(bbox1)):
                        occluded = False
                        for obj_name2 in ROAD_OBJECTS:
                            if obj_name1 == obj_name2: continue
                            bbox2 = boxes[obj_name2]
                            if bbox2 is None: continue
                            for j in range(len(bbox2)):
                                iou, mark = compute_iou(bbox1[i], bbox2[j])
                                if iou >= 0.9: 
                                    if OBJECTS_PRIORITY[obj_name1] > OBJECTS_PRIORITY[obj_name2]:
                                        occluded = False
                                    else:
                                        occluded = True
                        if occluded: continue      
                        tmp_boxes.append(bbox1[i])

                    if len(tmp_boxes) > 0:
                        boxes[obj_name1] = tmp_boxes
                        labels[obj_name1] = [labels[obj_name1][0]] * len(tmp_boxes)
                    else:
                        boxes[obj_name1] = None
                        labels[obj_name1] = None

        
                #  laneline segmentation
                # laneline_masks = laneline_segment(lane_model, img_path)
                # if isinstance(laneline_masks, torch.Tensor):
                #     laneline_masks = laneline_masks.cpu().numpy()
                #     boxes['laneline'] = [[0, 0, 1, 1]] * laneline_masks.shape[0]
                #     labels['laneline'] = ['laneline'] * laneline_masks.shape[0]
                
                init_segment = None
                for object_name in ROAD_OBJECTS:
                    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

                    input_boxes = boxes[object_name]
                    input_labels = labels[object_name]
                    if locations:
                        input_locations = locations[object_name]

                    masks = None
                    if input_boxes is not None and input_labels is not None:
                        tmp_boxes = []
                        for bbox in input_boxes:
                            x1, y1, x2, y2 = [max(0, coord) for coord in bbox]
                            x1 = min(width, x1)
                            x2 = min(width, x2)
                            y1 = min(height, y1)
                            y2 = min(height, y2)
                            tmp_boxes.append([x1, y1, x2, y2])
                        input_boxes = tmp_boxes

                        image_predictor.set_image(np.array(image.convert("RGB")))

                        if len(input_boxes) == 0:
                            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                            continue

                        masks, scores, logits = image_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )

                        # convert the mask shape to (n, H, W)
                        if masks.ndim == 2:
                            masks = masks[None]
                            scores = scores[None]
                            logits = logits[None]
                        elif masks.ndim == 4:
                            masks = masks.squeeze(1)

                    # If you are using point prompts, we uniformly sample positive points based on the mask
                    if mask_dict.promote_type == "mask" and masks is not None:
                        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=input_labels, location_list=input_locations)
                
                    objects_count[object_name] = mask_dict.update_masks(tracking_annotation_dict=sam2_masks[object_name], iou_threshold=0.80, objects_count=objects_count[object_name])
                    print("objects_count", objects_count[object_name])
                    video_predictor.reset_state(inference_state)
                    if len(mask_dict.labels) == 0:
                        print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                        continue
                    video_predictor.reset_state(inference_state)
                    video_segments, sam2_masks[object_name] = sam2_video_predictor(start_frame_idx, step, frame_names, mask_dict, sam2_masks[object_name], inference_state, video_predictor)
                    segments[object_name] = video_segments
                    if len(list(video_segments.keys())) > 0:
                        init_segment = video_segments

                if init_segment is None:
                    print(
                        "No new objects are detected in this frame!"
                    )
                    continue

            
                for frame_idx, _ in init_segment.items():
                    json_data, mask_image = {}, {}
                    for object_name in ROAD_OBJECTS:
                        video_segments = segments[object_name]
                        if frame_idx not in video_segments.keys():
                            json_data[object_name] = None
                            mask_image[object_name] = None
                            continue
                        frame_masks_info = video_segments[frame_idx]
                        mask = frame_masks_info.labels
                        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                        for obj_id, obj_info in mask.items():
                            mask_img[obj_info.mask == True] = obj_id

                        mask_image[object_name] = mask_img.numpy().astype(np.uint16)
                        json_data[object_name] = frame_masks_info.to_dict()

                        mask_name = frame_masks_info.mask_name.replace(".npy", ".pkl")
                        json_name = frame_masks_info.mask_name.replace(".npy", ".json")

                    with open(os.path.join(mask_data_dir, mask_name), "wb") as f:
                        pickle.dump(mask_image, f)

                    with open(os.path.join(json_data_dir, json_name), "w") as f:
                        f.write(json.dumps(json_data))

            CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)
            
            # plot lane segmentation
            image_name_list = os.listdir(result_dir)
            image_name_list.sort()
            json_name_list = os.listdir(json_data_dir)
            json_name_list.sort()
            lane_color = {}


            print(
                "Processing lane segmentation!"
            )
            for start_frame_idx in tqdm(range(len(frame_names))):
                if start_frame_idx >= len(samples): continue
                sample_token = samples[start_frame_idx]['sample_token']
                polygons, points, labels, map_name, ego_lane = loader._get_map_segmentation(sample_token, intersection_threshold=10000)
                polygon_dict = PolygonDictionaryModel()
                polygon_dict.add_new_frame_annotation(polygon_list=polygons, point_list=points, label_list=labels)
                lane_count = polygon_dict.update_polygons(tracking_annotation_dict=lane_polygon, iou_threshold=0.1, objects_count=lane_count)
                polygon_labels = polygon_dict.to_dict()['labels']

                
                image_path = os.path.join(result_dir, image_name_list[start_frame_idx])             
                image = Image.open(image_path)
    
                plt.figure(figsize=(image.size[0] / 100, image.size[1] / 100))
                plt.imshow(image)
                
                for lane_id, lane_info in polygon_labels.items():
                    if lane_id not in lane_color.keys():
                        random_color = random_color = (random.random(), random.random(), random.random())
                        lane_color[lane_id] = random_color
                    if lane_info['class_name'] == "ego":
                        ego_lane = str(lane_id)
                    lane_center_coords =  np.array(lane_info['polygon'].centroid.coords)
                    x, y = lane_center_coords[0][0], lane_center_coords[0][1]

                    plt.gca().add_patch(descartes.PolygonPatch(lane_info['polygon'], alpha=0.3, color=lane_color[lane_id], label="lane"))
                    plt.text(x, y, str(lane_id), color="white", fontsize=15, ha='center', va='center', 
                            bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))
                
                plt.axis('off')  
                plt.savefig(image_path, dpi=100, bbox_inches='tight', pad_inches=0)        
                plt.close()

                lane_polygon = polygon_dict
                samples[start_frame_idx]['lane_num'] = len(list(polygon_labels.keys()))
                json_data_name = "mask_" + image_name_list[start_frame_idx].split(".")[0] + ".json"
                
                # We need to save lane num
                if json_data_name in json_name_list:
                    json_data_path = os.path.join(json_data_dir, json_data_name)
                    with open(json_data_path, "r") as f:
                        json_data = json.load(f)
                else:
                    json_data_path = os.path.join(json_data_dir, json_data_name)
                    json_data = {}
               
                json_data['lane'] = polygon_dict.to_dict()
                for lane_id in json_data['lane']['labels'].keys():
                    json_data['lane']['labels'][lane_id]['polygon'] = ""

                json_data['lane']['lane_num'] = len(list(json_data['lane']['labels'].keys()))
                json_data['lane']['ego_lane'] = ego_lane
                
                with open(json_data_path, "w") as f:
                    f.write(json.dumps(json_data))
        
        print(
            f"Start to running VLM-Driver!"
        )

    
        vlm_driver(loader, samples, video_dir, result_dir, json_data_dir, using_mark)

        create_video_from_images(result_dir, output_video_path, frame_rate=15)

