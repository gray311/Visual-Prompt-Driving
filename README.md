# Visual Prompt Driving


### Install 
```
git clone https://github.com/gray311/Visual-Prompt-Driving.git
cd Visual-Prompt-Driving

conda create -n VPD python=3.10
conda activate VPD

pip3 install torch torchvision torchaudio

git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2

pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install einops flash_attn

pip install nuscenes-devkit
```

### Download dataset
```
mkdir -p ./workspace/nuscenes/
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xf v1.0-mini.tgz -C ./workspace/nuscenes/

mkdir -p ./workspace/checkpoint/
cd ./workspace/checkpoint/
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```
