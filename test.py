import pickle
from nuscenes.nuscenes import NuScenes
import os
import json
from collections import defaultdict



with open('/data/yingzi_ma/Visual-Prompt-Driving/workspace/uniad/nuscenes_infos_temporal_val.pkl','rb') as f:
    data = pickle.load(f)

print(data['infos'][0])

version = "v1.0-trainval"
dataroot = "/data/yingzi_ma/Visual-Prompt-Driving/workspace/nuscenes"
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)


sample_token2scene_id = {}
scene_id2sample_list = defaultdict(list)
for scene_id in range(850):
    try:
        scene = nusc.scene[scene_id]

    except:
        print(
            f"end of scene {scene_id}!"
        )
        break

    sample_token = scene['first_sample_token']
    while True:
        sample_token2scene_id[sample_token] = scene_id
        scene_id2sample_list[scene_id].append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']
        if sample_token == "":
            break

    
# print(sample_token2scene_id)

cnt = 0
scene_names = [
    # # 执行3点掉头
    # "scene-0778",

    # # 完全停止后恢复运动
    # "scene-0208", "scene-1023", "scene-0067", "scene-0159", "scene-0185",
    # "scene-0262", "scene-0862", "scene-0025", "scene-0072", "scene-0157",
    # "scene-0234", "scene-0423", "scene-0192", "scene-0657",
    # "scene-0921", "scene-0925", "scene-0968", "scene-0552", "scene-0917",
    # "scene-0221", "scene-1064", "scene-0331",

    # 超车通过对向车道
    "scene-0001", "scene-0011", "scene-0023", "scene-0034", "scene-0318",
    "scene-0379", "scene-0408", "scene-0417", "scene-0422", "scene-0865",
    "scene-1105", "scene-1065", "scene-0200", "scene-0752",
    "scene-0038", "scene-0271", "scene-0969", "scene-0329",

    # 绕过施工现场
    "scene-0980", "scene-0535"
]

def zero(): return 0
scene_id_num = defaultdict(zero)
for line in data['infos']:
    if line['token'] in sample_token2scene_id.keys():
        cnt += 1
        idx = sample_token2scene_id[line['token']]
        scene_id_num[idx] += 1

print(scene_id_num)
print(len(scene_id_num.keys()))
print(cnt)

val_scene_ids = [key for key in scene_id_num.keys()]

long_tail_ids = []
scene_name2scene_id = {}
for scene_id in range(850):
    scene = nusc.scene[scene_id]
    scene_name = scene['name']
    if scene_name in scene_names:
        long_tail_ids.append(scene_id)
        scene_name2scene_id[scene_name] = scene_id


scene_id_dict = {
    "train": [],
    "val": val_scene_ids,
    "long_tail": long_tail_ids,
}

print(long_tail_ids)

# with open("./dataset/scene_id.json", "w") as f:
#     f.write(json.dumps(scene_id_dict))


# with open("./dataset/long_tail_scene_name.json", "w") as f:
#     f.write(json.dumps(scene_name2scene_id))