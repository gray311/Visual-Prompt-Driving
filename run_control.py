import os
import re
import torch
from torch import Tensor
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import numpy as np
from typing import Optional
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from controller.MPC_fun import mpc_fun
from eval.metric import PlanningMetric

def convert_to_float(data):
    float_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            match = re.search(r'[-+]?\d*\.?\d+', value)
            if match:
                float_data[key] = float(match.group())
        elif isinstance(value, (int, float)):
            float_data[key] = float(value)
    return float_data

def print_testing_info(scene_name, filename):

    matches = re.findall(r'[sni](\d+)', file_name)
    numbers = list(map(int, matches))

    start_frame_idx, frame_num, frame_interval = numbers[0], numbers[1], numbers[2]
    current_frame_idx =  start_frame_idx + (frame_num - 1) * frame_interval
    input_frame_idx = range(start_frame_idx, current_frame_idx + 1, frame_interval)
    input_frame_str = [str(idx) for idx in input_frame_idx]
    input_frame_str = ", ".join(input_frame_str)

    print(
        f"Testing the {current_frame_idx}th frame in {scene_name} with historical inputs from the {input_frame_str} frames, predicting 8 future frames (4 seconds)."
    )



def proj_point_to_camera(nusc, point, sample_data, filter_min_dist=None):
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
     
    # Move box to ego vehicle coord system.
    # point += -np.array(pose_record['translation'])
    # point = np.dot(Quaternion(pose_record['rotation']).inverse.rotation_matrix, point)

    #  Move box to sensor coord system.
    point += -np.array(cs_record['translation'])
    point = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, point)

    depth_of_point = point[2]

    # see view_points(...), /nuscenes/utils/geometry_utils.py
    cam_m = np.eye(4)
    cam_m[0:3, 0:3] = cs_record["camera_intrinsic"]
    prj_point = np.dot(cam_m, np.array([point[0], point[1], point[2], 1]))
    prj_pimg = prj_point[0:2] / prj_point[2]

    if filter_min_dist is None:
        return prj_pimg
    if depth_of_point < filter_min_dist:
        return None
    return prj_pimg

def get_pred_fut_traj(result_path):
    with open(os.path.join(result_path, "prompt.txt"), "r") as f:
        prompt = f.read()

    with open(os.path.join(result_path, "response.txt"), "r") as f:
        response = f.read()

    mpc_start = response.find("### MPC")
    mpc_content = response[mpc_start:]
    # mpc_signals = re.findall(r'\*\*(.*?)\*\*: (.*?)\n', mpc_content)
    # if len(mpc_signals) == 0:
    #     mpc_signals = re.findall(r'\*\*(.*?):\*\* (.*?)\n', mpc_content)
    # mpc_dict = {re.sub(r'\s*\(.*?\)', '', key).strip(): value.strip() for key, value in mpc_signals}

    mpc_content = re.search(r'json\s*({.*?})', mpc_content, re.DOTALL).group(1)
    mpc_dict = json.loads(mpc_content)

    match = re.search(r'Speed \(m/s\): \[.*?([\d.]+)\s*\]', prompt)
    mpc_dict['speed'] = float(match.group(1))
    mpc_dict = convert_to_float(mpc_dict)

    v_ego = mpc_dict['speed']  # Initial speed in m/s
    N = 6  # Prediction horizon steps
    dt = 0.5  # Time step in seconds
    tau = 2.0  # Engine lag time constant
    
    mpc_list = [value for key, value in mpc_dict.items() if key != "speed"]
    mpc_list = mpc_list[:7]
    assert len(mpc_list) == 7, "lack of MPC control signals!"


    Q = mpc_list[0]  # Slightly increased speed maintenance weight for quicker accel/deceleration
    R = mpc_list[1]  # Slightly increased control effort weight for smoother accel/deceleration and turning
    Q_h = mpc_list[2]  # Headway maintenance weight (unchanged)
    desired_speed = mpc_list[3]  # Lower desired speed for aceel/deceleration
    desired_headway = mpc_list[4] # Desired headway in seconds
    desired_yaw_rate = mpc_list[5]  # Positive value for left turn (X-axis negative direction)
    yaw_rate_weight = mpc_list[6]  # Yaw rate maintenance weight

    lead_info = None  # No leading vehicle information

    # Run MPC with modified parameters
    d_values, v_values, u_values, delta_values = mpc_fun(v_ego, lead_info, N, dt, Q, R, Q_h, tau, desired_speed, desired_headway, desired_yaw_rate, yaw_rate_weight)

    # Analyze the new trajectory
    start_x, start_y = 0.0, 0.0
    trajectory = [(start_x, start_y)]
    for i in range(N):
        delta_x = -v_values[i] * dt * np.sin(delta_values[i])  # Left turn, x decreases
        delta_y = v_values[i] * dt * np.cos(delta_values[i])  # Forward motion, y increases
        new_x = trajectory[-1][0] + delta_x
        new_y = trajectory[-1][1] + delta_y
        trajectory.append([new_x[0], new_y[0]])
    
    trajectory = np.array(trajectory[1:])
    pred_fut_traj = np.expand_dims(trajectory, axis=0)
   
    return pred_fut_traj
    


if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-trainval", dataroot="/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes", verbose=True)


    scene_dir = "./images/baselines"

    with open('./eval/gt/vad_gt_seg.pkl','rb') as f:
        gt_occ_map_woP = pickle.load(f)
    for token in gt_occ_map_woP.keys():
        if not isinstance(gt_occ_map_woP[token], torch.Tensor):
            gt_occ_map_woP[token] = torch.tensor(gt_occ_map_woP[token])
        gt_occ_map_woP[token] = torch.flip(gt_occ_map_woP[token], [-1])
        gt_occ_map_woP[token] = torch.flip(gt_occ_map_woP[token], [-2])
    gt_occ_map = gt_occ_map_woP

    future_second = 3
    ts = future_second * 2
    device = torch.device('cpu')

    for scene_name in os.listdir(scene_dir):
        if ".jpg" in scene_name: continue
        if "nuscenes" not in scene_name: continue
        # if "scene_336"  not in scene_name: continue
        frame_path = os.path.join(scene_dir, scene_name)

        metric_planning_val = PlanningMetric(ts).to(device)  
        for file_name in os.listdir(frame_path):
            if ".jpg" in file_name: continue
            if ".pkl" in file_name: continue
            # if "s18" not in file_name: continue
            result_path = os.path.join(frame_path, file_name)

            print_testing_info(scene_name, file_name)

            pred_fut_traj = get_pred_fut_traj(result_path)

            with open(os.path.join(result_path, "gt_fut_traj.pkl"),'rb') as f:
                gt_trajectory = pickle.load(f)

            with open(os.path.join(result_path, "gt_fut_traj_mask.pkl"),'rb') as f:
                gt_traj_mask = pickle.load(f)
            
            sample_token =list(gt_trajectory.keys())[0]
            gt_trajectory = gt_trajectory[sample_token]
            gt_traj_mask = gt_traj_mask[sample_token]

            gt_trajectory = torch.from_numpy(gt_trajectory).to(device)
            gt_traj_mask = torch.from_numpy(gt_traj_mask).to(device)

            gt_trajectory = gt_trajectory[:, :6, :] # 3s in the future
            gt_traj_mask = gt_traj_mask[:, :6, :]

            output_trajs =  torch.from_numpy(pred_fut_traj).to(device)
            output_trajs = output_trajs.reshape(gt_traj_mask.shape)

            occupancy: Tensor = gt_occ_map[token]
            occupancy = occupancy.to(device)

            print("pred_traj: ", output_trajs)
            print("gt_traj: ", gt_trajectory)
            print("occupancy: ", occupancy.shape)

    
            with open(os.path.join(frame_path, file_name, "pred_fut_traj.pkl"), "wb") as f:
                pickle.dump({sample_token: output_trajs.numpy()}, f)

            if output_trajs.shape[1] % 2: # in case the current time is inculded
                output_trajs = output_trajs[:, 1:]

            if occupancy.shape[1] % 2: # in case the current time is inculded
                occupancy = occupancy[:, 1:]
            
            if gt_trajectory.shape[1] % 2: # in case the current time is inculded
                gt_trajectory = gt_trajectory[:, 1:]

            if gt_traj_mask.shape[1] % 2:  # in case the current time is inculded
                gt_traj_mask = gt_traj_mask[:, 1:]

            metric_planning_val(output_trajs[:, :ts], gt_trajectory[:, :ts], occupancy[:, :ts], file_name, gt_traj_mask) 
        

    print(
        f"\n\n\nTesting in {scene_name}!"
    )
    results = {}
    scores = metric_planning_val.compute()
    for i in range(future_second):
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)]=value[:(i+1)*2].mean()

    # Print results in table
    print(f"gt collision: {metric_planning_val.gt_collision}")
    headers = ["Method", "L2 (m)", "Collision (%)"]
    sub_headers = ["1s", "2s", "3s", "Avg."]
    method = ("VisualPrompt", "{:.2f}".format(scores["L2"][1]), "{:.2f}".format(scores["L2"][3]), "{:.2f}".format(scores["L2"][5]),\
            "{:.2f}".format((scores["L2"][1]+ scores["L2"][3]+ scores["L2"][5]) / 3.),
            "{:.2f}".format(scores["obj_box_col"][1]*100), \
            "{:.2f}".format(scores["obj_box_col"][3]*100), \
            "{:.2f}".format(scores["obj_box_col"][5]*100), \
            "{:.2f}".format(100*(scores["obj_box_col"][1]+ scores["obj_box_col"][3]+ scores["obj_box_col"][5]) / 3.))
    print("\n")
    print("UniAD evaluation:")
    print("{:<15} {:<20} {:<20}".format(*headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))

    method = ("VisualPrompt", "{:.2f}".format(results["plan_L2_1s"]), "{:.2f}".format(results["plan_L2_2s"]), "{:.2f}".format(results["plan_L2_3s"]), \
                "{:.2f}".format((results["plan_L2_1s"]+results["plan_L2_2s"]+results["plan_L2_3s"])/3.), 
                "{:.2f}".format(results["plan_obj_box_col_1s"]*100), "{:.2f}".format(results["plan_obj_box_col_2s"]*100), "{:.2f}".format(results["plan_obj_box_col_3s"]*100), \
                    "{:.2f}".format(((results["plan_obj_box_col_1s"] + results["plan_obj_box_col_2s"] + results["plan_obj_box_col_3s"])/3)*100)) 
    print("\n")
    print("STP-3 evaluation:")
    print("{:<15} {:<20} {:<20}".format(*headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))


    


            




            

