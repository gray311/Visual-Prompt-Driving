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
from dataset.nuscenes_dataset import NuscenesLoader
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from ultralytics import YOLO


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


def build_model(sam2_checkpoint, model_cfg):
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

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    lane_model = YOLO("./workspace/checkpoint/yolov8_seg.pt")

    return image_predictor, video_predictor, grounding_model, lane_model, processor, device

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


def get_lane_masks(model, image_path, mode="yolo", threshold=200, plot_laneline=True):
    results = model(image_path, conf=0.3)

    for result in results:
        boxes = result.boxes  
        masks = result.masks 
    
    if masks is None:
        return None, None, None
 
    width, height = masks.shape[1:]
    laneline_xy = sorted(masks.xy, key=(lambda x: np.sum(x, axis=0)[0] / x.shape[0]), reverse=False)
    laneline_xy = [np.array([[0.0, height/2.0]])] + laneline_xy + [np.array([[width, height/2.0]])]
    

    # def mark_bool(): return defaultdict(bool)
    # def mark_list(): return defaultdict(list)
    # lane_mark = defaultdict(mark_bool)
    # lane = defaultdict(mark_list)
    # flag = False
    # for i in range(len(laneline_xy)):
    #     laneline_xy[i] = laneline_xy[i][laneline_xy[i][:, 1] <= 700]
    #     laneline_xy[i] = laneline_xy[i][laneline_xy[i][:, 1] >= 400]
    #     if laneline_xy[i].shape[0] == 0: continue
    #     lane_left_x = np.sum(laneline_xy[i], axis=0)[0] / laneline_xy[i].shape[0]
    #     lane_left_y = np.sum(laneline_xy[i], axis=1)[0] / laneline_xy[i].shape[1]
    #     for j in range(i+1, len(laneline_xy)):
    #         if lane_mark[i][j] == True: break
    #         laneline_xy[j] = laneline_xy[j][laneline_xy[j][:, 1] <= 700]
    #         laneline_xy[j] = laneline_xy[j][laneline_xy[j][:, 1] >= 400]
    #         if laneline_xy[j].shape[0] == 0: continue
    #         lane_right_x = np.sum(laneline_xy[j], axis=0)[0] / laneline_xy[j].shape[0]
    #         lane_right_y = np.sum(laneline_xy[j], axis=1)[0] / laneline_xy[j].shape[1]
    #         if lane_right_x - lane_left_x >= threshold:
    #             x, y = (lane_left_x + lane_right_x) / 2.0, (lane_left_y + lane_right_y) / 2.0
    #             if j == len(laneline_xy) - 1: 
    #                 if flag == False:
    #                     flag = True
    #                     lane[i][j] = [lane_left_x + 150, 600 - 50, x + 250, 600 + 50]
    #                     lane_mark[i][j] = True
    #             elif i == 0:
    #                 lane[i][j] = [lane_right_x - 250, 600 - 50, lane_right_x - 150, 600 + 50]
    #                 lane_mark[i][j] = True
    #             else:
    #                 lane[i][j] = [x - 50, 600 - 50, x + 50, 600 + 50]
    #                 lane_mark[i][j] = True
    #             break

    # lane_boxes = []
    # for i in range(len(laneline_xy)):
    #     for j in range(i+1, len(laneline_xy)):
    #         if len(lane[i][j]) > 0:
    #             lane_boxes.append(np.array(lane[i][j]))

    # remove_idx = []
    # for i in range(len(lane_boxes)):
    #     for j in range(i+1, len(lane_boxes)): 
    #         if bounding_boxes_close(lane_boxes[i], lane_boxes[j]):
    #             remove_idx.append(j)
    
    # for idx in remove_idx:
    #     del lane_boxes[idx]

    # lane_number_masks = bounding_box_to_mask((width, height), lane_boxes)

    lane_masks = F.interpolate(masks.data.unsqueeze(1), size=masks.orig_shape, mode='nearest').squeeze(1)


    return lane_masks, None, None


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

def concat_image(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, height1))

    return new_image


def run_agent(samples, video_dir, result_dir):

    frame_num, frame_interval = 4, 2 # get history 4 frames from 0s, sampling frequency 1s.
    frame_span = (frame_num - 1) * frame_interval

    video_files = get_filename(video_dir)
    result_files = get_filename(result_dir)

    start_frame_list = [i * frame_span  for i in range(len(video_files) // frame_span)] # No need to predict every frame

    for start_frame_idx in start_frame_list:
        frames, marks, ego_states = [], [], []
        output_dir = os.path.join(video_dir, f"s{start_frame_idx}_n{frame_num}_i{frame_interval}")
        if not os.path.exists(output_dir):
            # shutil.rmtree(output_dir)
            os.mkdir(output_dir)

        for idx, filename in enumerate(video_files):
            if idx >= start_frame_idx and (idx - start_frame_idx) % frame_interval == 0 and (idx - start_frame_idx) / frame_interval < frame_num: 
                frames.append(Image.open(os.path.join(video_dir, filename)))
                marks.append(Image.open(os.path.join(result_dir, result_files[idx])))
                ego_states.append((samples[idx]["ego_vehicle_velocity"], samples[idx]["ego_vehicle_accelerate"], samples[idx]["ego_vehicle_heading"]))
                
                if idx - start_frame_idx == (frame_num - 1) * frame_interval:
                    ego_future_trajectory = samples[idx]['ego_future_trajectory']
                    ego_future_trajectory_mask = samples[idx]['ego_future_trajectory_mask']
                    token = samples[idx]['sample_token']
                

        image_list = []
        for frame, mark in zip(frames, marks):
            image = concat_image(frame, mark)
            image.save(f"./outputs/{len(image_list)}.png")
            image_list.append(image)

        from api import GeminiEvaluator, GPTEvaluator, system_message, user_message

        model_name = "gpt"
        if model_name == "gemini":
            agent = GeminiEvaluator(api_key="")
        elif model_name == "gpt":
            agent = GPTEvaluator(api_key="")
    
        # navigation_instruction = "Please follow the car in front of you."
        navigation_instruction = "The right lane looks clear. Can you change to the right lane?"
        ego_vel_str = "[" + ", ".join([str(round(item[0], 2)) for item in ego_states]) + "]"
        ego_accel_str = "[" + ", ".join([str(round(item[1], 2)) for item in ego_states]) + "]"
        ego_angle_str = "[" + ", ".join([str(round(item[2], 2)) for item in ego_states]) + "]"

        prompt = user_message.format(speed=ego_vel_str, acceleration=ego_accel_str, angle=ego_angle_str)
        
        if navigation_instruction:
            prompt += "\n\n" + f"3. Navigation command: {navigation_instruction}"
        prompt += "\n\nOutput:"

        question = {
            "prompted_system_content": system_message,
            "prompted_content": prompt,
            "image_list": image_list,
        }

        # if not os.path.exists(os.path.join(output_dir, "response.txt")):
        #     response = agent.generate_answer(question)

        #     with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        #         f.write(prompt)

        #     with open(os.path.join(output_dir, "response.txt"), "w") as f:
        #         f.write(response['prediction'])
        # else:
        #     print(
        #         f"MPC control signals have been generated by VLM!"
        #     )

        response = agent.generate_answer(question)

        with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        with open(os.path.join(output_dir, "response.txt"), "w") as f:
            f.write(response['prediction'])

        with open(os.path.join(output_dir, "gt_fut_traj.pkl"), "wb") as f:
            pickle.dump({token: np.array(ego_future_trajectory)}, f)
        
        with open(os.path.join(output_dir, "gt_fut_traj_mask.pkl"), "wb") as f:
            pickle.dump({token: np.array(ego_future_trajectory_mask)}, f)


if __name__=="__main__":
    scene_id_file = "./dataset/scene_id.json"
    sam2_checkpoint = "./workspace/checkpoint/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    image_predictor, video_predictor, grounding_model, lane_model, processor, device = build_model(sam2_checkpoint, model_cfg)

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

        sam2_masks = MaskDictionaryModel()
        PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
        objects_count = 0
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
            mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

            if use_ground_truth:
                instances = samples[start_frame_idx]['sample_annotations']
                remove_id = []
                OBJECTS = [item['category_name'].split("/")[-1] for instance_id, item in instances.items() if instance_id not in remove_id]
                input_boxes = [item['bounding_box'] for instance_id, item in instances.items() if instance_id not in remove_id]
                input_boxes = torch.tensor(input_boxes)
            else:
                # run Grounding DINO on the image
                # we can use driving expert model to detect bounding box or lane segmentation
                input_boxes, OBJECTS = [], [] 
                for text in ['car, and bus.']:
                    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = grounding_model(**inputs)

                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=0.25,
                        text_threshold=0.25,
                        target_sizes=[image.size[::-1]]
                    )

                    input_boxes.extend(results[0]["boxes"].cpu().numpy().tolist())
                    OBJECTS.extend(results[0]["labels"])

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

            # use YoloV8 to detect lane line
            lane_masks, lane_number_masks, lane_boxes = get_lane_masks(lane_model, img_path)
            
            if lane_masks is not None:
                lane_masks = lane_masks.cpu().numpy()
                masks = np.concatenate((masks, lane_masks), axis=0)
                input_boxes = input_boxes + input_boxes[-lane_masks.shape[0]:]
                OBJECTS = OBJECTS + ['laneline'] * lane_masks.shape[0]

            # If you are using point prompts, we uniformly sample positive points based on the mask
            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")

            objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.80, objects_count=objects_count)
            print("objects_count", objects_count)
            video_predictor.reset_state(inference_state)
            if len(mask_dict.labels) == 0:
                print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                continue
            video_predictor.reset_state(inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )
            
            video_segments = {}  # output the following {step} frames tracking masks
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)


            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id

                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)


        CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)
        
        run_agent(samples, video_dir, result_dir)

        create_video_from_images(result_dir, output_video_path, frame_rate=15)
    







