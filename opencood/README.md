We use the [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) framework to evaluate our UrbanIng-V2X dataset on cooperative perception models.

# Setup

## Clone the repository
```bash
git clone https://github.com/thi-ad/UrbanIng-V2X.git
cd UrbanIng-V2X
```

## Create conda environment
```bash
conda env create -f environment.yml
conda activate UrbanIng_v2x
```

## PyTorch installation
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## Install other dependencies
```bash
pip install -r requirements.txt
python setup.py develop
```

## spconv 2.x Installation
```bash
pip install spconv-cu113
```

## Install bbx nms calculation cuda version
```bash
python opencood/utils/setup.py build_ext --inplace
```


# Training 
```bash
python opencood/tools/train.py --hypes_yaml hypes_yaml/<placeholder>.yaml --half
```


# Inference
```bash
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
