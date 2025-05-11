For simplicity, please create two separate environments for "scripts" and "opencood".


##### SCRIPTS #####

In scripts, you find a "requirements.txt" to create your environment with the needed packages.
We used python version 3.10

# Visualize the dataset with open3d visualizer
python scripts/open3d_visualizer.py  # please choose the dataset folder, labels folder and sequence in the code
Note: Use keys '1', '2', 'A', 'D', '3', '4' to navigate within the frames ; Use key 'E' to change colors between intensity, lidar source and time offset

# Visualize the dataset by creating videos
python scripts/video_creation.py  # please choose the dataset folder, labels folder, save folder and sequence in the code


##### OPENCOOD #####

### PREPARE DATASET STRUCTURE ###
Convert the original dataset format to OPENCOOD format using python scripts.

1) Using 'scripts/create_opencood_format.py' and setting your paths pointing to the downloaded dataset creates the OPENCOOD format.
2) To create the same splits as they are used in our experiments, please run 'scripts/create_opencood_splits.py'


### INSTALLATION ###
# Create conda environment (python == 3.9)
conda create -n v2x_dataset python=3.9 -y
conda activate v2x_dataset

# pytorch installation
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install other dependencies
pip install -r requirements.txt
python setup.py develop

# spconv 2.x Installation
pip install spconv-cu113

# Install bbx nms calculation cuda version
python opencood/utils/setup.py build_ext --inplace


### TRAINING ###

python opencood/tools/train.py --hypes_yaml hypes_yaml/xxxx.yaml --half


### INFERENCE ###

python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]

