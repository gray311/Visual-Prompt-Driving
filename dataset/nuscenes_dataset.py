import nuscenes
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, Quaternion
from nuscenes.utils.geometry_utils import box_in_image
from .utils import get_project_root, relative_heading_angle, get_rotation_matrix
import numpy as np
DEFAULT_SENSOR_CHANNEL = 'CAM_FRONT'


def get_prj_matrix(pose_record, cs_record):
    t1 = -np.array(pose_record['translation'])
    pose_rotation = pose_record['rotation']
    R1 = Quaternion(pose_rotation).inverse.rotation_matrix
    # R (x + t) = R x + t'. You want this t' ? If so, it will be t' = R t .
    t1 = np.dot(R1, t1)
    E1 = np.column_stack((R1, t1))

    t2 = -np.array(cs_record['translation'])
    cs_rotation = cs_record['rotation']
    R2 = Quaternion(cs_rotation).inverse.rotation_matrix
    # R (x + t) = R x + t'. You want this t' ? If so, it will be t' = R t .
    t2 = np.dot(R2, t2)
    E2 = np.column_stack((R2, t2))
    cs_camera_intrinsic = cs_record['camera_intrinsic']

    # The combined transformation matrix (E) can be obtained by multiplying E2 with E1
    E = np.dot(np.row_stack((E2, [0, 0, 0, 1])), np.row_stack((E1, [0, 0, 0, 1])))
    prj_mat = np.dot(cs_camera_intrinsic, E[0:3, :])

    return prj_mat


class NuscenesLoader:
    def __init__(self, version, dataroot, frequency=5):
        # project_root = get_project_root()
        # dataroot = os.path.join(project_root, 'data', 'nuscenes')
        self.dataroot = dataroot
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.frequency = frequency

    def get_scene_description(self, scene_id):
        scene = self.nusc.scene[scene_id]
        scene_description = scene['description']
        number_of_samples = scene['nbr_samples']
        first_sample_token = scene['first_sample_token']
        start_time = self.nusc.get('sample',scene['first_sample_token'])['timestamp']
        end_time = self.nusc.get('sample',scene['last_sample_token'])['timestamp']
        full_description = {
            'scene_description': scene_description,
            'number_of_samples': number_of_samples,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        return full_description, first_sample_token

    def get_reference_pose(self, sample_token):
        # ego vehicle is the vehicle in which the camera is mounted
        sample = self.nusc.get('sample', sample_token)
        ego_pose_token = self.nusc.get('sample_data', sample['data'][DEFAULT_SENSOR_CHANNEL])['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        ego_vehicle_pose = {
            'translation': ego_pose['translation'],
            'rotation': ego_pose['rotation']
        }
        ego_rotation_matrix = get_rotation_matrix(ego_vehicle_pose['rotation'])
        return ego_vehicle_pose, ego_rotation_matrix

    def get_sample_description(self, sample_token):
        # each sample is a frame in the scene consisting of a set of annotations in different sensors

        sample = self.nusc.get('sample', sample_token)
        ego_pose_token = self.nusc.get('sample_data', sample['data'][DEFAULT_SENSOR_CHANNEL])['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        timestamp = sample['timestamp']
        visible_annotations_by_channel, filepath_by_channel = self.get_annotations_for_sensors(sample)
        anns = visible_annotations_by_channel[DEFAULT_SENSOR_CHANNEL]
        filepath = filepath_by_channel[DEFAULT_SENSOR_CHANNEL]
        object_descriptions = self.get_object_description(anns)
        location, heading = self.transform_global_to_local(
                ego_pose,
                self.reference_pose,
                self.reference_rotation_matrix
        )
        # print(location,heading)
        sample_description = {
            'ego_vehicle_location': location,
            'ego_vehicle_heading': heading,
            'timestamp': timestamp,
            'filepath': filepath,
            'sample_annotations': object_descriptions
        }
        next_sample_token = sample['next']
        return sample_description, next_sample_token

    def get_annotations_for_sensors(self, sample, sensor_channels=[DEFAULT_SENSOR_CHANNEL]):
        visible_annotations_by_channel = {} 
        filepath_by_channel = {}

        for sensor_channel in sensor_channels:
            # Get the sample data token for the specified sensor channel
            sample_data_token = sample['data'][sensor_channel]
            # Get the sample data
            sd_record = self.nusc.get('sample_data', sample_data_token)
            filepath = os.path.join(self.nusc.dataroot, sd_record['filename'])
            cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
            
            prj_mat = get_prj_matrix(pose_record, cs_record)
            visible_annotations = []
            annotation_tokens = sample['anns']
            for ann_token in annotation_tokens:
                # Get the annotation data
                ann = self.nusc.get('sample_annotation', ann_token)
                # Create a Box instance for the annotation
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']),name=ann['category_name'], token=ann['token'])
                
                cc = np.copy(box.center)
                vet2 = np.array([[cc[0]], [cc[1]], [cc[2]], [1]])
                temp_c = np.dot(prj_mat, vet2)
                point_3d2img = temp_c[0:2] / temp_c[2]
    
                # Move box to ego vehicle coord system
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                if box_in_image(box, cam_intrinsic, imsize, vis_level=1):
                    ann['center_point'] = [point_3d2img[0][0], point_3d2img[1][0]]
                    ann['instance_token'] = ann_token
                    visible_annotations.append(ann)
            visible_annotations_by_channel[sensor_channel] = visible_annotations
            filepath_by_channel[sensor_channel] = filepath

        return visible_annotations_by_channel, filepath_by_channel

    def get_object_description(self, anns):

        print(anns)

        object_descriptions = {}
        for i, ann_metadata in enumerate(anns):

            object_description = {}
            # ann_token = sample['anns'][i]
            # ann_metadata =  self.nusc.get('sample_annotation', ann_token)
            visibility = self.nusc.get('visibility', ann_metadata['visibility_token'])['level']
            size = ann_metadata['size']
            if ann_metadata['attribute_tokens']:
                attribute = self.nusc.get('attribute', ann_metadata['attribute_tokens'][0])['name']
                object_description['attribute'] = attribute
            name = ann_metadata['category_name']
            instance_token = ann_metadata['instance_token']
            center_point = ann_metadata['center_point']

            object_description['visibility'] = visibility
            object_description['size_in_meter'] = size
            object_description['category_name'] = name
            object_description['center_point'] = center_point
            object_description['instance_token'] = instance_token

            location, heading = self.transform_global_to_local(
                ann_metadata,
                self.reference_pose,
                self.reference_rotation_matrix
            )
            object_description['location'] = location
            if heading:
                object_description['heading'] = heading
            object_descriptions[i] = object_description
        return object_descriptions

    def transform_global_to_local(self, ann_metadata, reference_pose, rotation_matrix):
        heading = None
        if ann_metadata['rotation']:
            heading = relative_heading_angle(
                ann_metadata['rotation'],
                reference_pose['rotation']
            )
        
        location = (np.around(
                        np.dot(rotation_matrix,np.array(ann_metadata['translation'])) - np.dot(rotation_matrix,np.array(reference_pose['translation'])),
                        2
                        )
                    ).tolist()
        return location, heading

    def load(self, scene_id):
        # use scene_id as the token for scene cannot be easily got
        # the frequency is the number of samples to skip

        scene_description, first_sample_token = self.get_scene_description(scene_id)
        sample_descriptions = {}
        index, sample_token = 0, first_sample_token
        self.reference_pose, self.reference_rotation_matrix = self.get_reference_pose(sample_token)

        while True:
            descriptions, next_sample_token = self.get_sample_description(sample_token)
            if index % self.frequency == 0:
                sample_descriptions[index] = descriptions
            index += 1
            if next_sample_token == '':
                break
            sample_token = next_sample_token

        metadata = {
            'scene_description': scene_description,
            'sample_descriptions': sample_descriptions
        }
        return metadata

if __name__ == "__main__":
    ## Test the function
    loader = NuscenesLoader(version="v1.0-mini", dataroot="/home/yingzi/Visual-Prompt-Driving/workspace/nuscenes")
    metadata = loader.load(1)
    print(metadata)