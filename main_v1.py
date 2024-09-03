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
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers import AutoProcessor, AutoModelForCausalLM
from dataset.nuscenes_dataset import NuscenesLoader
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from ultralytics import YOLO
from experts import *

ROAD_OBJECTS = ['car.', 'bus.', 'truck.', 'pedestrain.']


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


def generate_realtime_data(scene_id, loader):
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
    
    output_dir = f"./outputs/nuscenes_scene_{scene_id}"
    if not os.path.exists(output_dir):
        # shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    # 'output_video_path' is the path to save the final video
    output_video_path = f"./outputs/nuscenes_scene_{scene_id}/output.mp4"

    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    frame_names = get_filename(video_path)

    return samples, video_path, output_dir, output_video_path, mask_data_dir, json_data_dir, result_dir, frame_names



if __name__=="__main__":
    scene_id_file = "./dataset/scene_id.json"
    sam2_checkpoint = "./workspace/checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    grounding_model_name = "dino"
    image_predictor, video_predictor, grounding_model, lane_model, processor, device = build_model(sam2_checkpoint, model_cfg, grounding_model_name=grounding_model_name)

    loader = NuscenesLoader(version="v1.0-mini", dataroot="/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes", frequency=1)
    
    with open(scene_id_file, "r") as f:
        scene_id_split = json.load(f)

    overtake_scene_ids = [1]
    
    from tqdm import tqdm
    for i in tqdm(overtake_scene_ids):
        samples, video_dir, output_dir, output_video_path, \
        mask_data_dir, json_data_dir, result_dir, frame_names  = generate_realtime_data(scene_id=i, loader=loader)

        # init video predictor state
        inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
        step = 4 # the step to sample frames for Grounding DINO predictor

        sam2_masks = {object_name: MaskDictionaryModel() for object_name in ROAD_OBJECTS}
        objects_count = {object_name: 0 for object_name in ROAD_OBJECTS}
        PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
        use_ground_truth = False

        # if has_image_files(result_dir):
        #     run_agent(samples, video_dir, result_dir)
        #     continue

        print(f"Frame num: {len(frame_names)}")
        for start_frame_idx in range(0, len(frame_names), step):
            print("start_frame_idx", start_frame_idx)
            # continue
            img_path = os.path.join(video_dir, frame_names[start_frame_idx])
            image = Image.open(img_path)
            image_base_name = frame_names[start_frame_idx].split(".")[0]
        
            # if use_ground_truth:
            #     instances = samples[start_frame_idx]['sample_annotations']
            #     remove_id = []
            #     OBJECTS = [item['category_name'].split("/")[-1] for instance_id, item in instances.items() if instance_id not in remove_id]
            #     input_boxes = [item['bounding_box'] for instance_id, item in instances.items() if instance_id not in remove_id]
            #     input_boxes = torch.tensor(input_boxes)
           
            # run Grounding DINO on the image
            # we can use driving expert model to detect bounding box or lane segmentation
            boxes, labels = defaultdict(list), defaultdict(list) 
            segments = defaultdict(dict)
            OBJECTS_NUM = {object_name: 0 for object_name in ROAD_OBJECTS}

            if grounding_model_name == "dino":
                for object_name in ROAD_OBJECTS:
                    input_boxes, input_labels = dino_detect_object(image, object_name, grounding_model, processor)
                    boxes[object_name] = input_boxes
                    labels[object_name] = input_labels
                    OBJECTS_NUM[object_name] = len(input_labels)
            elif grounding_model_name == "florence2":
                boxes, labels = florence2_detect_object(image, "car <and> truck <and> pedestrain <and> bus", grounding_model, processor)
         
            laneline_masks = laneline_segment(lane_model, img_path)
            if laneline_masks is not None:
                laneline_masks = laneline_masks.cpu().numpy()
                boxes['laneline'] = input_boxes[0][0] * laneline_masks.shape[0]
                labels['laneline'] = ['laneline'] * laneline_masks.shape[0]
            

            for object_name in ROAD_OBJECTS:
                mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

                input_boxes = boxes[object_name]
                input_labels = labels[object_name]

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

                if laneline_masks is not None and input_labels[0] == "laneline":
                    masks = laneline_masks

                # If you are using point prompts, we uniformly sample positive points based on the mask
                if mask_dict.promote_type == "mask":
                    mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=input_labels)
                else:
                    raise NotImplementedError("SAM 2 video predictor only support mask prompts")

                objects_count[object_name] = mask_dict.update_masks(tracking_annotation_dict=sam2_masks[object_name], iou_threshold=0.80, objects_count=objects_count[object_name])
                print("objects_count", objects_count[object_name])
                video_predictor.reset_state(inference_state)
                if len(mask_dict.labels) == 0:
                    print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                    continue
                video_predictor.reset_state(inference_state)

                video_segments, sam2_masks[object_name] = sam2_video_predictor(start_frame_idx, step, frame_names, mask_dict, sam2_masks[object_name], inference_state, video_predictor)
                
                segments[object_name] = video_segments
         
            for object_name in ROAD_OBJECTS:
                video_segments = segments[object_name]
                for frame_idx, frame_masks_info in video_segments.items():
                    mask = frame_masks_info.labels
                    mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                    for obj_id, obj_info in mask.items():
                        mask_img[obj_info.mask == True] = obj_id

                    mask_img = mask_img.numpy().astype(np.uint16)
                    print(frame_masks_info.to_dict())
                    np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

                    json_data = frame_masks_info.to_dict()
                    json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                    with open(json_data_path, "w") as f:
                        json.dump(json_data, f)

        import sys

        sys.exit(0)

        CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)
        
        vlm_driver(samples, video_dir, result_dir)

        # create_video_from_images(result_dir, output_video_path, frame_rate=15)
    







