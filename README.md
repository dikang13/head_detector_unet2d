# UNet2D for Object Detection

This repository provides a 2D U-Net-based pipeline for segmenting fluorescent microscopy images. It is designed for images where object boundaries are unclear or ill-defined. My specific use case concerns the anterior pharynx of the nematode C. elegans, the outline of which is not labelled by fluorescent markers in my input images.


## Features
- **2D U-Net Architecture**: Well-suited for biomedical image segmentation.
- **Configurable Data Augmentation**: Hyperparameters are specified in `unet-2d-head-detector-train.yaml` for reproducible experiments and easy adaptation.
- **Checkpointing and Logging**: Saves model checkpoints and training logs for easy recovery and visualization.


## Data Requirements

- **Fluorescent Microscopy Images**:  
  These typically have unclear boundaries that require robust segmentation methods.

- **Mask Format**:  
  Binary or multi-class masks. The code expects masks to be in the same resolution and shape as the corresponding images.

- **Combined File Format**:  
  Images and Masks must be in identical dimensions. I precompiled them into HDF5 forma here for portability. Code for data preparation can be found under `notebooks/prepare_data`. Data used to train, validate and test my model are included in `data`.


## Installation

To ensure a reproducible environment, a Conda environment file (`environment.yaml`) is provided.

1. **Clone the repository** (or download the source code):
   ```bash
   git clone git@github.com:flavell-lab/head_detector_unet2d.git
   cd head_detector_unet2d
   ```

2. **Create the Conda environment**:
   ```bash
   conda env create -f environment.yaml
   ```

3. **Activate the environment**:
   ```bash
   conda activate head_detector_unet2d
   ```

4. **Install unet2d** (submodule cloned from `git@github.com:flavell-lab/unet2d.git`):
   ```bash
   cd unet2d
   pip install -e .
   ```

5. **Verify installation**:
   ```bash
   conda list
   # or
   python -c "import torch; import imgaug; print('Installation successful!')"
   ```

## Usage

### Training

1. **Edit the YAML config**  
   Open `unet2d-head-detector-train.yaml`to modify training hyperparameters and augmentation settings.

2. **Run training**:
   ```bash
   python src/train.py \
        --config ./config.yaml \
        --output ./exp \
        --device cuda:2
   ```
     
3. **Examine loss and performance curves**:
   ```bash
    python src/plot_training_curves.py \
        --csv_path ./exp/unet2d-20250303/unet2d-20250303.csv \
        --output_path ./exp/unet2d-20250303/loss_curve.png
   ```   

### Inference

1. **Run inference**:
   ```bash
   python src/inference.py \
        --model ./exp/unet2d-20250303/unet2d-20250303_best.pt \
        --config ./config.yaml \
        --output ./exp/unet2d-20250303/predictions \
        --device cuda:2
   ```

2. **Examine model predictions**:
   Dice scores for each test image from model inference is saved as a `.csv` in `./exp/(exp_name)`. For objects without boundaries (such as the anterior pharynx of the worm), it if often helpful to download the output png folder `./exp/(exp_name)/predictions`, which contains png files of images, ground truth and model prediction, to your local machine and visually examine if the prediction is on par with your expectation.


## Troubleshooting
    
- **Out of Memory Errors**  
  Try reducing `batch_size` in `unet-2d-head-detector-train.yaml`.

- **Slow Training**  
  - Ensure you have GPU acceleration (CUDA installed).
  - The pipeline was tested on a single NVIDIA RTX A5500 and 6000 Ada. Using configurations specified in `unet-2d-head-detector-train.yaml`, model training finished in 3 hours and model inference on 500 test images took 3 minutes.
  - Heavier augmentation is expected to lengthen training time. 