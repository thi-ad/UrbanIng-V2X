# UrbanIng-V2X: A Large-Scale Multi-Vehicle, Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception

This package provides tools for dataset download, extraction, conversion, handling and visualization for the UrbanIng-V2X dataset.

## Install via PyPI
```bash
pip install urbaning
```

## Downloading the dataset
```python
from urbaning.data import download_dataset, download_one_sequence
download_dataset(download_dir="datasets/UrbanIng-V2X") # to download the entire dataset
download_one_sequence(download_dir="datasets/UrbanIng-V2X") # to download only one sequence for quick start purposes - optionally pass a sequence_name
```

## Unzip the dataset
Note: 7zip has to be installed.
```python
from urbaning.data import unzip_dataset
unzip_dataset(dataset_folder="datasets/UrbanIng-V2X")
# If 7zip is not visible in the environment, pass also the sevenz_executable parameter
```

## Dataset structure
```yaml
. [DATA_ROOT] # Dataset root folder
├── 📂dataset # data files
│   ├── 📂20241126_0001_crossing2_00 # sequence 1's data
│   │   ├── 📂crossing2_11_lidar # this and upcoming folders -> infrastructure lidars
│   │   │   ├── 🌫️1732632673950.npz # point cloud captured by crossing2_11_lidar at this time stamp
│   │   │   └   ...
│   │   ├── 📂crossing2_12_lidar
│   │   ├── 📂crossing2_31_lidar
│   │   ├── 📂crossing2_32_lidar
│   │   ├── 📂crossing2_13_thermal_camera # this and upcoming folders -> infrastructure cameras
│   │   │   ├── 🖼️1732632673956.jpg # image captured by crossing2_13_thermal_camera at this time stamp
│   │   │   └   ...
│   │   ├── 📂crossing2_14_thermal_camera 
│   │   ├── 📂crossing2_15_thermal_camera
│   │   ├── 📂crossing2_33_thermal_camera
│   │   ├── 📂crossing2_34_thermal_camera
│   │   ├── 📂vehicle1_back_left_camera # this and upcoming folders -> vehicle1 cameras
│   │   │   ├── 🖼️1732632674019.jpg # image captured by vehicle1_back_left_camera at this time stamp
│   │   │   └   ...
│   │   ├── 📂vehicle1_back_right_camera
│   │   ├── 📂vehicle1_front_left_camera
│   │   ├── 📂vehicle1_front_right_camera
│   │   ├── 📂vehicle1_left_camera
│   │   ├── 📂vehicle1_right_camera
│   │   ├── 📂vehicle1_middle_lidar # vehicle1 lidar
│   │   │   ├── 🌫️1732632673950.npz # point cloud captured by vehicle1_middle_lidar at this time stamp
│   │   │   └   ...
│   │   ├── 📂vehicle1_state
│   │   │   ├── 🚘1732632670000.json # state information of vehicle1 at this time stamp
│   │   │   └   ...
│   │   ├── 📂vehicle2_back_left_camera # this and upcoming folders -> vehicle2 cameras
│   │   │   ├── 🖼️1732632674019.jpg # image captured by vehicle2_back_left_camera at this time stamp
│   │   │   └   ...
│   │   ├── 📂vehicle2_back_right_camera
│   │   ├── 📂vehicle2_front_left_camera
│   │   ├── 📂vehicle2_front_right_camera
│   │   ├── 📂vehicle2_left_camera
│   │   ├── 📂vehicle2_right_camera
│   │   ├── 📂vehicle2_middle_lidar # vehicle2 lidar
│   │   │   ├── 🌫️1732632673950.npz # point cloud captured by vehicle2_middle_lidar at this time stamp
│   │   │   └   ...
│   │   ├── 📂vehicle2_state
│   │   │   ├── 🚘1732632670000.json # state information of vehicle2 at this time stamp
│   │   │   └   ...
│   │   ├── 🧭calibration.json # all intrinsic and extrinsic calibration parameters for both vehicles and infrastructures
│   │   ├── 📊timesync_info.csv # time synchronization information linking several sensor data together 
│   │   └── 📄weather_data.json # weather_data during the data collection
│   ├── 📂20241126_0004_crossing2_00
│   ├── 📂20241126_0008_crossing1_00
│   └   ...
├── 📂labels # label files
│   ├── 📄20241126_0001_crossing2_00.json # sequence 1's labels
│   ├── 📄20241126_0004_crossing2_00.json
│   ├── 📄20241126_0008_crossing1_00.json
│   └   ...
├── 📂digital_twin # carla digital twin folder
├── 📄av_vehicle_data.json # static details like track width, axle length for connected vehicles
├── 📄crossings_lanelet2map.osm # HD Lanelet map of the crossings
└── 📄labels_av_track_ids.json # track IDs of connected vehicles in the labels
```


## Accessing the dataset
```python
from urbaning import Dataset
# root folder where the dataset is downloaded and unzipped
root_folder = "datasets/UrbanIng-V2X"
# load the complete dataset
dataset = Dataset(root_folder)
# number of total sequences
print(len(dataset))
```

## Accessing a sequence
```python
# use indexing from the dataset
index = 10
# a Sequence instance
print(type(dataset[index]))
# or simple iterate over the dataset
for sequence in dataset:
    print(sequence.sequence_name)
    break

# you can also load one sequence manually
from urbaning.data import Sequence
# sequence to load
sequence_name = "20241126_0017_crossing1_00"
# load the sequence
sequence = Sequence(root_folder, sequence_name)
```

## Accessing a frame
```python
# use indexing from the sequence
index = 115
# a Frame instance
print(type(sequence[index]))
# or simple iterate over the sequence
for frame in sequence:
    # this frame timestamp
    print(frame.timestamp)
    break
```

For more information on how to access the dataset, check out the [tutorial.ipynb](https://github.com/thi-ad/UrbanIng-V2X/blob/main/tutorial.ipynb) file.