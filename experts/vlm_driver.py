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
from api import GeminiEvaluator, GPTEvaluator, system_message, user_message
from experts.utils import get_filename


def prepare_vision_input(frame_num, frame_interval, samples, video_files, result_files):
    frames, marks, ego_states = [], [], []
    for idx, filename in enumerate(video_files):
        if idx >= start_frame_idx and (idx - start_frame_idx) % frame_interval == 0 and (idx - start_frame_idx) / frame_interval < frame_num: 
            frames.append(Image.open(os.path.join(video_dir, filename)))
            marks.append(Image.open(os.path.join(result_dir, result_files[idx])))
            ego_states.append((samples[idx]["ego_vehicle_velocity"], samples[idx]["ego_vehicle_accelerate"], samples[idx]["ego_vehicle_heading"]))
            if idx - start_frame_idx == (frame_num - 1) * frame_interval:
                last_frame_idx = idx
            
    image_list = []
    for frame, mark in zip(frames, marks):
        image = concat_image(frame, mark)
        image.save(f"./outputs/{len(image_list)}.png")
        image_list.append(image)

    return image_list, ego_states, last_frame_idx

def prepare_text_input(navigation_instruction, ego_states):
    ego_vel_str = "[" + ", ".join([str(round(item[0], 2)) for item in ego_states]) + "]"
    ego_accel_str = "[" + ", ".join([str(round(item[1], 2)) for item in ego_states]) + "]"
    ego_angle_str = "[" + ", ".join([str(round(item[2], 2)) for item in ego_states]) + "]"

    prompt = user_message.format(speed=ego_vel_str, acceleration=ego_accel_str, angle=ego_angle_str)
    
    if navigation_instruction:
        prompt += "\n\n" + f"3. Navigation command: {navigation_instruction}"
    prompt += "\n\nOutput:"

    return prompt


def vlm_driver(samples, video_dir, result_dir):
    frame_num, frame_interval = 4, 2 # get history 4 frames from 0s, sampling frequency 1s.
    frame_span = (frame_num - 1) * frame_interval
    start_frame_list = [i * frame_span  for i in range(len(video_files) // frame_span)] # No need to predict every frame

    video_files = get_filename(video_dir)
    result_files = get_filename(result_dir)
    
    for start_frame_idx in start_frame_list:
        output_dir = os.path.join(video_dir, f"s{start_frame_idx}_n{frame_num}_i{frame_interval}")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        image_list, ego_states, last_frame_idx = prepare_vision_input(frame_num, frame_interval, samples, video_files, result_files)
        
        # navigation_instruction = "Please follow the car in front of you."
        navigation_instruction = "The right lane looks clear. Can you change to the right lane?"
        prompt = prepare_text_input(navigation_instruction, ego_states)
        
        
        model_name = "gpt"
        if model_name == "gemini":
            agent = GeminiEvaluator(api_key="")
        elif model_name == "gpt":
            agent = GPTEvaluator(api_key="")
    
        question = {
            "prompted_system_content": system_message,
            "prompted_content": prompt,
            "image_list": image_list,
        }


        response = agent.generate_answer(question)

        ego_future_trajectory = samples[last_frame_idx]['ego_future_trajectory']
        ego_future_trajectory_mask = samples[last_frame_idx]['ego_future_trajectory_mask']
        token = samples[last_frame_idx]['sample_token']

        with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        with open(os.path.join(output_dir, "response.txt"), "w") as f:
            f.write(response['prediction'])

        with open(os.path.join(output_dir, "gt_fut_traj.pkl"), "wb") as f:
            pickle.dump({token: np.array(ego_future_trajectory)}, f)
        
        with open(os.path.join(output_dir, "gt_fut_traj_mask.pkl"), "wb") as f:
            pickle.dump({token: np.array(ego_future_trajectory_mask)}, f)
