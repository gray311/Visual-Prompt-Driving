import os
import numpy as np
import pickle
from pathlib import Path
import json
import ast
import re
from PIL import Image, ImageDraw, ImageFont

from cam_render import CameraRender
from utils import AgentPredictionData, color_mapping

from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
import imageio

def draw_raw(sample_token, nusc):
    cam_render = CameraRender(show_gt_boxes=False)

    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)

    save_path = Path("experiments/visualization") / Path(sample_token + '_raw.jpg')
    cam_render.save_fig(save_path)

def draw_inputs(sample_token, nusc, samples):
    cam_render = CameraRender(show_gt_boxes=False)
    det_obj_color = np.array([51, 153,255]) / 255.0
    planner_input = None
    for sample in samples:
        if sample['token'] == sample_token:
            planner_input = sample
            break
    
    if planner_input is None:
        return

    agent_list = []
    data_dict = pickle.load(open('data/val/'+sample_token+'.pkl', 'rb'))["objects"]
    num_object = len(data_dict)
    for i in range(num_object):
        pred_boxes = data_dict[i]["bbox"]
        pred_center=pred_boxes[:3]
        pred_dim=pred_boxes[3:6]
        pred_center[2] += pred_dim[2]/2
        pred_traj = data_dict[i]["traj"][:6,:]
        pred_traj = np.concatenate([pred_traj, np.zeros((6,1))], axis=-1)
        name = data_dict[i]["name"]
        print(f"obj {name}: center: ({pred_center[0]:.2f}, {pred_center[1]:.2f}), size: ({pred_dim[0]:.2f}, {pred_dim[1]:.2f}, {pred_dim[1]:.2f})")
        print(f"traj: [({pred_traj[0,0]:.2f}, {pred_traj[0,1]:.2f}), ({pred_traj[1,0]:.2f}, {pred_traj[1,1]:.2f}), ({pred_traj[2,0]:.2f}, {pred_traj[2,1]:.2f}), ({pred_traj[3,0]:.2f}, {pred_traj[3,1]:.2f}), ({pred_traj[4,0]:.2f}, {pred_traj[4,1]:.2f}), ({pred_traj[5,0]:.2f}, {pred_traj[5,1]:.2f})]")
        agent_list.append(
            AgentPredictionData(
                pred_score=1.0,
                pred_label=0,
                pred_center=pred_center,
                pred_dim=pred_dim,
                pred_yaw=pred_boxes[6],
                pred_vel=0,
                pred_traj=pred_traj,
                is_sdc = False
            )
        )

    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)
    cam_render.render_pred_track_bbox(
        agent_list, sample_token, nusc, box_color=det_obj_color)
    cam_render.render_pred_traj(
        agent_list, sample_token, nusc, sdc_color=color_mapping[1], render_sdc=False, box_color=det_obj_color)

    save_path = Path("experiments/visualization") / Path(sample_token + '_input.jpg')
    cam_render.save_fig(save_path)


def draw_tools(sample_token, nusc, samples):
    cam_render = CameraRender(show_gt_boxes=False)
    tool_obj_color = np.array([255, 153, 51]) / 255.0
    planner_input = None
    for sample in samples:
        if sample['token'] == sample_token:
            planner_input = sample
            break
    
    if planner_input is None:
        return

    agent_list = []
    data_dict = pickle.load(open('data/val/'+sample_token+'.pkl', 'rb'))["objects"]
    num_object = len(data_dict)
    for i in range(num_object):
        pred_boxes = data_dict[i]["bbox"]
        pred_center=pred_boxes[:3]
        pred_dim=pred_boxes[3:6]
        pred_center[2] += pred_dim[2]/2
        pred_traj = data_dict[i]["traj"][:6,:]
        pred_traj = np.concatenate([pred_traj, np.zeros((6,1))], axis=-1)
        name = data_dict[i]["name"]
        obj_x, obj_y = pred_center[:2]
        if abs(obj_x) < 10.0 and obj_y >= 0.0 and obj_y <= 40.0:
            print(f"obj {name}: center: ({pred_center[0]:.2f}, {pred_center[1]:.2f}), size: ({pred_dim[0]:.2f}, {pred_dim[1]:.2f}, {pred_dim[1]:.2f})")
            print(f"traj: [({pred_traj[0,0]:.2f}, {pred_traj[0,1]:.2f}), ({pred_traj[1,0]:.2f}, {pred_traj[1,1]:.2f}), ({pred_traj[2,0]:.2f}, {pred_traj[2,1]:.2f}), ({pred_traj[3,0]:.2f}, {pred_traj[3,1]:.2f}), ({pred_traj[4,0]:.2f}, {pred_traj[4,1]:.2f}), ({pred_traj[5,0]:.2f}, {pred_traj[5,1]:.2f})]")
            agent_list.append(
                AgentPredictionData(
                    pred_score=1.0,
                    pred_label=0,
                    pred_center=pred_center,
                    pred_dim=pred_dim,
                    pred_yaw=pred_boxes[6],
                    pred_vel=0,
                    pred_traj=pred_traj,
                    is_sdc = False
                )
            )
    box_color = np.array([255, 128, 0]) / 255.0
    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)
    cam_render.render_pred_track_bbox(
        agent_list, sample_token, nusc, box_color=tool_obj_color)
    cam_render.render_pred_traj(
        agent_list, sample_token, nusc, sdc_color=color_mapping[1], render_sdc=False, box_color=tool_obj_color)

    save_path = Path("experiments/visualization") / Path(sample_token + '_tools.jpg')
    cam_render.save_fig(save_path)

def draw_cot(sample_token, nusc, samples):
    cam_render = CameraRender(show_gt_boxes=False)
    cot_obj_color = np.array([255, 51, 51]) / 255.0
    
    planner_input = None
    for sample in samples:
        if sample['token'] == sample_token:
            planner_input = sample
            break
    
    if planner_input is None:
        return

    reasoning = planner_input['reasoning']
    print(reasoning)
    notable_objects = []
    notable_coords = []
    if 'None' in reasoning:
        pass
    else:
        for line in reasoning.split("\n"):
            if "Notable Objects:" in line:
                terms = line.strip().split(' ')
                print(terms)
                try:
                    notable_objects.append(terms[3])
                    coord = re.search(r"\(-?\d+\.\d+, ?-?\d+\.\d+\)",line)
                    notable_coords.append(ast.literal_eval(coord.group()))
                    # if terms[5][:-1] == ',':
                    #     notable_coords.append(ast.literal_eval(terms[5][:-1]))
                    # else:
                    #     notable_coords.append(ast.literal_eval(terms[5]))
                except:
                    if len(notable_objects) > len(notable_coords):
                        notable_objects.pop()
                    continue

        print(notable_objects)
        print(notable_coords)


    agent_list = []
    data_dict = pickle.load(open('data/val/'+sample_token+'.pkl', 'rb'))["objects"]
    num_object = len(data_dict)
    for i in range(num_object):
        pred_boxes = data_dict[i]["bbox"]
        pred_center=pred_boxes[:3]
        pred_dim=pred_boxes[3:6]
        pred_center[2] += pred_dim[2]/2
        pred_traj = data_dict[i]["traj"][:6,:]
        pred_traj = np.concatenate([pred_traj, np.zeros((6,1))], axis=-1)
        for j, notable_object in enumerate(notable_objects):
            notable_coord = notable_coords[j]
            dist = pred_center[:2] - np.array(notable_coord) 
            if np.linalg.norm(dist) < 6.0:
                print(f"{notable_object}: {notable_coord}")
                agent_list.append(
                    AgentPredictionData(
                        pred_score=1.0,
                        pred_label=0,
                        pred_center=pred_center,
                        pred_dim=pred_dim,
                        pred_yaw=pred_boxes[6],
                        pred_vel=0,
                        pred_traj=pred_traj,
                        is_sdc = False
                    )
                )
    box_color = np.array([255, 0, 0]) / 255.0
    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)
    cam_render.render_pred_track_bbox(
        agent_list, sample_token, nusc, box_color=cot_obj_color)
    cam_render.render_pred_traj(
        agent_list, sample_token, nusc, sdc_color=color_mapping[1], render_sdc=False, box_color=cot_obj_color)

    save_path = Path("experiments/visualization") / Path(sample_token + '_ureason.jpg')
    cam_render.save_fig(save_path)

def draw_plan(sample_token, nusc, samples, plan_trajs_dict, gt_trajs_dict):
    sdc_pred_color = np.array([255, 51, 51]) / 255.0
    sdc_gt_color = np.array([51, 255, 51]) / 255.0
    
    cam_render = CameraRender(show_gt_boxes=False)

    planner_input = None
    for sample in samples:
        if sample['token'] == sample_token:
            planner_input = sample
            break
    
    if planner_input is None:
        return

    reasoning = planner_input['reasoning']
    print(reasoning)
    notable_objects = []
    notable_coords = []
    if 'None' in reasoning:
        pass
    else:
        for line in reasoning.split("\n"):
            if "Notable Objects:" in line:
                terms = line.strip().split(' ')
                print(terms)
                try:
                    notable_objects.append(terms[3])
                    coord = re.search(r"\(-?\d+\.\d+, ?-?\d+\.\d+\)",line)
                    notable_coords.append(ast.literal_eval(coord.group()))
                    # if terms[5][:-1] == ',':
                    #     notable_coords.append(ast.literal_eval(terms[5][:-1]))
                    # else:
                    #     notable_coords.append(ast.literal_eval(terms[5]))
                except:
                    if len(notable_objects) > len(notable_coords):
                        notable_objects.pop()
                    continue

        print(notable_objects)
        print(notable_coords)

    agent_list = []
    data_dict = pickle.load(open('data/val/'+sample_token+'.pkl', 'rb'))["objects"]
    num_object = len(data_dict)
    for i in range(num_object):
        pred_boxes = data_dict[i]["bbox"]
        pred_center=pred_boxes[:3]
        pred_dim=pred_boxes[3:6]
        pred_center[2] += pred_dim[2]/2
        pred_traj = data_dict[i]["traj"][:6,:]
        pred_traj = np.concatenate([pred_traj, np.zeros((6,1))], axis=-1)
        for j, notable_object in enumerate(notable_objects):
            notable_coord = notable_coords[j]
            dist = pred_center[:2] - np.array(notable_coord) 
            if np.linalg.norm(dist) < 6.0:
                print(f"{notable_object}: {notable_coord}")
                agent_list.append(
                    AgentPredictionData(
                        pred_score=1.0,
                        pred_label=0,
                        pred_center=pred_center,
                        pred_dim=pred_dim,
                        pred_yaw=pred_boxes[6],
                        pred_vel=0,
                        pred_traj=pred_traj,
                        is_sdc = False
                    )
                )
    box_color = np.array([255, 0, 0]) / 255.0
    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)
    cam_render.render_pred_track_bbox(
        agent_list, sample_token, nusc, box_color=box_color)
    cam_render.render_pred_traj(
        agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=False, box_color=box_color)


    if sample_token not in plan_trajs_dict:
        return
    plan_traj = plan_trajs_dict[sample_token]
    plan_traj = np.concatenate([plan_traj, np.ones((6,1))], axis=-1)

    if sample_token not in gt_trajs_dict:
        return
    gt_traj = gt_trajs_dict[sample_token]['gt_trajectory']
    gt_traj = np.array(gt_traj)[1:]

    gt_agent_list = [
        AgentPredictionData(
        pred_score=1.0,
        pred_label=0,
        pred_center=[0, 0, 0],
        pred_dim=[4.5, 2.0, 2.0],
        pred_yaw=0,
        pred_vel=0,
        pred_traj=gt_traj,
        is_sdc = True
        )
    ]
    cam_render.render_pred_traj(
        gt_agent_list, sample_token, nusc, sdc_color=sdc_gt_color, render_sdc=True)


    pred_agent_list = [
        AgentPredictionData(
            pred_score=1.0,
            pred_label=0,
            pred_center=[0, 0, 0],
            pred_dim=[4.5, 2.0, 2.0],
            pred_yaw=0,
            pred_vel=0,
            pred_traj=plan_traj,
            is_sdc = True
        )
    ]
    cam_render.render_pred_traj(
        pred_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=True)

    save_path = Path("experiments/visualization") / Path(sample_token + '_xplan.jpg')
    cam_render.save_fig(save_path)


def draw_all(sample_token, nusc, samples, plan_trajs_dict, gt_trajs_dict):
    sdc_pred_color = np.array([255, 51, 51]) / 255.0
    sdc_gt_color = np.array([51, 255, 51]) / 255.0
    det_obj_color = np.array([51, 153,255]) / 255.0
    tool_obj_color = np.array([255, 153, 51]) / 255.0
    cot_obj_color = np.array([255, 51, 51]) / 255.0
    

    cam_render = CameraRender(show_gt_boxes=False)

    planner_input = None
    for sample in samples:
        if sample['token'] == sample_token:
            planner_input = sample
            break
    
    if planner_input is None:
        return

    reasoning = planner_input['reasoning']
    print(reasoning)
    notable_objects = []
    notable_coords = []
    if 'None' in reasoning:
        pass
    else:
        for line in reasoning.split("\n"):
            if "Notable Objects:" in line:
                terms = line.strip().split(' ')
                print(terms)
                try:
                    notable_objects.append(terms[3])
                    coord = re.search(r"\(-?\d+\.\d+, ?-?\d+\.\d+\)",line)
                    notable_coords.append(ast.literal_eval(coord.group()))
                    # if terms[5][:-1] == ',':
                    #     notable_coords.append(ast.literal_eval(terms[5][:-1]))
                    # else:
                    #     notable_coords.append(ast.literal_eval(terms[5]))
                except:
                    if len(notable_objects) > len(notable_coords):
                        notable_objects.pop()
                    continue

        print(notable_objects)
        print(notable_coords)

    # if len(notable_objects) != len(notable_coords):
        # return

    det_agent_list, tool_agent_list, cot_agent_list = [], [], []
    data_dict = pickle.load(open('data/val/'+sample_token+'.pkl', 'rb'))["objects"]
    num_object = len(data_dict)
    is_cot_obj = False
    for i in range(num_object):
        is_cot_obj = False
        pred_boxes = data_dict[i]["bbox"]
        pred_center=pred_boxes[:3]
        pred_dim=pred_boxes[3:6]
        pred_center[2] += pred_dim[2]/2
        pred_traj = data_dict[i]["traj"][:6,:]
        pred_traj = np.concatenate([pred_traj, np.zeros((6,1))], axis=-1)
        for j, notable_object in enumerate(notable_objects):
            notable_coord = notable_coords[j]
            dist = pred_center[:2] - np.array(notable_coord) 
            if np.linalg.norm(dist) < 6.0:
                print(f"{notable_object}: {notable_coord}")
                cot_agent_list.append(
                    AgentPredictionData(
                        pred_score=1.0,
                        pred_label=0,
                        pred_center=pred_center,
                        pred_dim=pred_dim,
                        pred_yaw=pred_boxes[6],
                        pred_vel=0,
                        pred_traj=pred_traj,
                        is_sdc = False
                    )
                )
                is_cot_obj = True
        if is_cot_obj:
            continue
        else:
            obj_x, obj_y = pred_center[:2]
            if abs(obj_x) < 10.0 and obj_y >= 0.0 and obj_y <= 40.0:
                tool_agent_list.append(
                    AgentPredictionData(
                        pred_score=1.0,
                        pred_label=0,
                        pred_center=pred_center,
                        pred_dim=pred_dim,
                        pred_yaw=pred_boxes[6],
                        pred_vel=0,
                        pred_traj=pred_traj,
                        is_sdc = False
                    )
                )
            else:
                det_agent_list.append(
                    AgentPredictionData(
                        pred_score=1.0,
                        pred_label=0,
                        pred_center=pred_center,
                        pred_dim=pred_dim,
                        pred_yaw=pred_boxes[6],
                        pred_vel=0,
                        pred_traj=pred_traj,
                        is_sdc = False
                    )
                )
    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)
    
    cam_render.render_pred_track_bbox(
        cot_agent_list, sample_token, nusc, box_color=cot_obj_color)
    cam_render.render_pred_traj(
        cot_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=False, box_color=cot_obj_color)
    cam_render.render_pred_track_bbox(
        tool_agent_list, sample_token, nusc, box_color=tool_obj_color)
    cam_render.render_pred_traj(
        tool_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=False, box_color=tool_obj_color)
    cam_render.render_pred_track_bbox(
        det_agent_list, sample_token, nusc, box_color=det_obj_color)
    cam_render.render_pred_traj(
        det_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=False, box_color=det_obj_color)

    if sample_token not in plan_trajs_dict:
        return
    plan_traj = plan_trajs_dict[sample_token]
    plan_traj = np.concatenate([plan_traj, np.ones((6,1))], axis=-1)

    if sample_token not in gt_trajs_dict:
        return
    gt_traj = gt_trajs_dict[sample_token]['gt_trajectory']
    gt_traj = np.array(gt_traj)[1:]

    gt_agent_list = [
        AgentPredictionData(
        pred_score=1.0,
        pred_label=0,
        pred_center=[0, 0, 0],
        pred_dim=[4.5, 2.0, 2.0],
        pred_yaw=0,
        pred_vel=0,
        pred_traj=gt_traj,
        is_sdc = True
        )
    ]
    cam_render.render_pred_traj(
        gt_agent_list, sample_token, nusc, sdc_color=sdc_gt_color, render_sdc=True)

    pred_agent_list = [
        AgentPredictionData(
            pred_score=1.0,
            pred_label=0,
            pred_center=[0, 0, 0],
            pred_dim=[4.5, 2.0, 2.0],
            pred_yaw=0,
            pred_vel=0,
            pred_traj=plan_traj,
            is_sdc = True
        )
    ]
    cam_render.render_pred_traj(
        pred_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=True)

    save_path = Path("experiments/visualization") / Path(sample_token + '_all.jpg')
    cam_render.save_fig(save_path)

def draw_text(sample_token, nusc, response, result_path, plan_trajs_dict, gt_trajs_dict, baseline_traj_dict=None):
    sdc_pred_color = np.array([255, 51, 51]) / 255.0
    sdc_baseline_color = np.array([51, 51, 255]) / 255.0
    sdc_gt_color = np.array([51, 255, 51]) / 255.0
    
   
    cam_render = CameraRender(show_gt_boxes=False)
    cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
    cam_render.render_image_data(sample_token, nusc)

    if sample_token not in plan_trajs_dict.keys():
        return
    plan_traj = plan_trajs_dict[sample_token]
    plan_traj = np.concatenate([plan_traj[0], np.ones((6,1))], axis=-1)

    if baseline_traj_dict is not None:
        if sample_token not in baseline_traj_dict.keys():
            return
        baseline_traj = baseline_traj_dict[sample_token]
        baseline_traj = np.concatenate([baseline_traj[0], np.ones((6,1))], axis=-1)

    if sample_token not in gt_trajs_dict.keys():
        return
    gt_traj = gt_trajs_dict[sample_token]
    gt_traj = np.array(gt_traj)
    gt_traj = np.concatenate([gt_traj[0], np.ones((12,1))], axis=-1)

    gt_agent_list = [
        AgentPredictionData(
        pred_score=1.0,
        pred_label=0,
        pred_center=[0, 0, 0],
        pred_dim=[4.5, 2.0, 2.0],
        pred_yaw=0,
        pred_vel=0,
        pred_traj=gt_traj,
        is_sdc = True
        )
    ]
    cam_render.render_pred_traj(
        gt_agent_list, sample_token, nusc, sdc_color=sdc_gt_color, render_sdc=True)

    pred_agent_list = [
        AgentPredictionData(
            pred_score=1.0,
            pred_label=0,
            pred_center=[0, 0, 0],
            pred_dim=[4.5, 2.0, 2.0],
            pred_yaw=0,
            pred_vel=0,
            pred_traj=plan_traj,
            is_sdc = True
        )
    ]
    cam_render.render_pred_traj(
        pred_agent_list, sample_token, nusc, sdc_color=sdc_pred_color, render_sdc=True)

    if baseline_traj_dict is not None:
        baseline_agent_list = [
            AgentPredictionData(
                pred_score=1.0,
                pred_label=0,
                pred_center=[0, 0, 0],
                pred_dim=[4.5, 2.0, 2.0],
                pred_yaw=0,
                pred_vel=0,
                pred_traj=baseline_traj,
                is_sdc = True
            )
        ]
        cam_render.render_pred_traj(
            baseline_agent_list, sample_token, nusc, sdc_color=sdc_baseline_color, render_sdc=True)

    save_path = os.path.join(result_path, sample_token + ".jpg")
    cam_render.save_fig(save_path)

    print(
        f"ego planning trajectory has been visulizated in {save_path}!"
    )
    
    # import pdb; pdb.set_trace()
    # reasoning = planner_input['reasoning']
    # reasoning = reasoning.split("\n")[:-1]
    # cot = planner_input['chain_of_thoughts']
    # cot = [cot.split("\n")[-2]]
    # planning = planner_input['planning_target']
    # planning = [planning.split("\n")[-1]]
    # text = ""
    # for line in reasoning:
    #     text += line + "\n"
    # text += "*****Task Planning:*****:\n"
    # for line in cot:
    #     text += line + "\n"
    # text += "*****Motion Planning:*****:\n"
    # for line in planning:
    #     text += line + "\n"
    # text += "*****Self-Reflection:*****:\n"
    # text += "No collision\n"
    # cat_text(save_path, text)

def create_text_image(text, img_size=(5333, 800)):
    image = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', size=50)
    # font = ImageFont.load_default()
    x = 0
    y = 0
    draw.text((x, y), text, font=font, fill='black')
    
    return image

def cat_text(save_path, text):
    image1 = Image.open(save_path)
    image2 = create_text_image(text)

    new_image = Image.new('RGB', (5333, 2800), (255, 255, 255))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, 2000))

    new_image.save(save_path)
    return

def make_gif(sample_token):
    print(sample_token)
    file_names = []
    file_names.append(Path("experiments/visualization") / Path(sample_token + '_input.jpg'))
    file_names.append(Path("experiments/visualization") / Path(sample_token + '_tools.jpg'))
    file_names.append(Path("experiments/visualization") / Path(sample_token + '_ureason.jpg'))
    file_names.append(Path("experiments/visualization") / Path(sample_token + '_xplan.jpg'))
    
    images = []
    for filename in file_names:
        images.append(imageio.imread(filename))
    imageio.mimsave(Path("experiments/visualization") / Path(sample_token + '.gif'), images, duration=1.0)  # Save the images as a GIF
    return

def print_visualization_info(scene_name, filename):

    matches = re.findall(r'[sni](\d+)', file_name)
    numbers = list(map(int, matches))

    start_frame_idx, frame_num, frame_interval = numbers[0], numbers[1], numbers[2]
    current_frame_idx =  start_frame_idx + (frame_num - 1) * frame_interval
    input_frame_idx = range(start_frame_idx, current_frame_idx + 1, frame_interval)
    input_frame_str = [str(idx) for idx in input_frame_idx]
    input_frame_str = ", ".join(input_frame_str)

    print(
        f"Visualize the {current_frame_idx}th frame in {scene_name} with historical inputs from the {input_frame_str} frames, predicting 8 future frames (4 seconds)."
    )

if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-trainval", dataroot="/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes", verbose=True)
    # plan_trajs_dict = pickle.load(open('pred_trajs_dict.pkl', 'rb')) # your pred traj dict here
    # gt_trajs_dict = pickle.load(open('data/metrics/gt_traj.pkl', 'rb'))

    scene_dir = "./images"
    for scene_name in os.listdir(scene_dir):
        if ".jpg" in scene_name: continue
        if "nuscenes_scene" not in scene_name: continue

        frame_path = os.path.join(scene_dir, scene_name)
        for file_name in os.listdir(frame_path):
            if ".jpg" in file_name: continue
            if ".pkl" in file_name: continue
            # if "s18" not in file_name: continue
            result_path = os.path.join(frame_path, file_name)
            baseline_path = os.path.join(scene_dir, "baseline", scene_name, file_name)

            print_visualization_info(scene_name, file_name)

            with open(os.path.join(result_path, "gt_fut_traj.pkl"),'rb') as f:
                gt_trajectory = pickle.load(f)

            with open(os.path.join(result_path, "gt_fut_traj_mask.pkl"),'rb') as f:
                gt_traj_mask = pickle.load(f)

            with open(os.path.join(result_path, "pred_fut_traj.pkl"),'rb') as f:
                pred_trajectory = pickle.load(f)
            
            if os.path.exists(os.path.join(baseline_path, "pred_fut_traj.pkl")):
                with open(os.path.join(baseline_path, "pred_fut_traj.pkl"),'rb') as f:
                    baseline_trajectory = pickle.load(f)

            with open(os.path.join(result_path, "response.txt"), "rb") as f:
                response = f.read()

            sample_token =list(gt_trajectory.keys())[0]
            draw_text(sample_token, nusc, response, result_path, pred_trajectory, gt_trajectory, baseline_trajectory)

 