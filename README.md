# 🚗 Urbaning-V2X Dataset Preparation & Visualization

This repository provides tools for dataset visualization, conversion, and training using **OpenCOOD**.  
For simplicity, two separate environments are used: one for **scripts** and one for **OpenCOOD**.

---

## 📁 Environments Overview

| Environment | Description |
|--------------|-------------|
| **scripts**  | Used for visualization and dataset conversion (Python 3.10). |
| **opencood** | Used for training and inference using the OpenCOOD framework (Python 3.9). |

---

### 1️⃣ Scripts Environment

```bash
# Create environment
conda create -n scripts python=3.10 -y
conda activate scripts

# Install dependencies
pip install -r scripts/requirements.txt
```

### 2️⃣ OpenCOOD Environment
```bash
# Create environment
conda create -n v2x_dataset python=3.9 -y
conda activate v2x_dataset

# Install PyTorch
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install other dependencies
pip install -r requirements.txt
python setup.py develop

# Install spconv 2.x
pip install spconv-cu113

# Build CUDA NMS extension
python opencood/utils/setup.py build_ext --inplace
```

## ⚙️ Scripts Usage

### 1️⃣ Visualize Dataset with Open3D
```
python scripts/open3d_visualizer.py
``` 
- Set the dataset folder, labels folder, and sequence in the script.
- Navigation keys:
- '1', '2', 'A', 'D', '3', '4' → Navigate frames
- 'E' → Switch color modes (intensity, lidar source, time offset)

### 2️⃣ Visualize Dataset as Video
```
python scripts/video_creation.py
```

- Define dataset folder, labels folder, save folder, and sequence in the script.

### 3️⃣ nuScenes Format Conversion
```
python scripts/create_nuscenes_format.py
```

- Set your dataset paths in the script.

- Define train/validation/test splits in the main() function.

- Configure ego_agents to specify ego agent(s).

### 4️⃣ OpenCOOD Format Conversion

- Convert original dataset to OpenCOOD format:

```
python scripts/create_opencood_format.py
```

- Create dataset splits matching experiment settings:

```
python scripts/create_opencood_splits.py
```

(Use the scripts environment for these steps.)

## 🏋️ Training OpenCOOD
```
python opencood/tools/train.py --hypes_yaml hypes_yaml/xxxx.yaml --half
```
- Replace xxxx.yaml with your configuration file.
- `--half`:  Enables mixed precision training (optional).

## 🔍 Inference OpenCOOD
```
python opencood/tools/inference.py \
  --model_dir ${CHECKPOINT_FOLDER} \
  --fusion_method ${FUSION_STRATEGY} \
  [--show_vis] [--show_sequence]
```

- `--show_vis`: Enable visualization
- `--show_sequence`: Show sequential frame results

## 🧾 Summary of Scripts & Commands

| Task | Script / Command | Environment |
|------|-----------------|-------------|
| Dataset visualization (Open3D) | `python scripts/open3d_visualizer.py` | scripts |
| Video creation | `python scripts/video_creation.py` | scripts |
| nuScenes format conversion | `python scripts/create_nuscenes_format.py` | scripts |
| OpenCOOD format conversion | `python scripts/create_opencood_format.py` | scripts |
| Create OpenCOOD splits | `python scripts/create_opencood_splits.py` | scripts |
| Model training | `python opencood/tools/train.py` | opencood |
| Inference | `python opencood/tools/inference.py` | opencood |

💡 Tips

- Keep dataset paths absolute to avoid errors.

- Ensure CUDA is properly installed for training and NMS compilation.

- Use separate environments for scripts and OpenCOOD to prevent dependency conflicts.