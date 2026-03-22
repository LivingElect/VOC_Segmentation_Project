# VOC Semantic Segmentation Project

This project implements a semantic segmentation model for the PASCAL VOC 2012 dataset using a Fully Convolutional U-Net architecture.

## Project Structure

```
VOC_Segmentation_Project/
├── data/                        # Data directory
│   └── VOCdevkit/VOC2012/      # VOC 2012 dataset
│       ├── JPEGImages/         # Input images
│       └── SegmentationClass/  # Pixel-level segmentation masks
├── configs/                     # Configuration files
│   └── voc_seg.yaml            # VOC segmentation configuration
├── datasets/                    # Dataset classes
│   └── voc_seg_dataset.py      # VOC segmentation dataset
├── models/                      # Model architectures
│   └── fcn_unet.py             # Fully Convolutional U-Net
├── utils/                       # Utility functions
│   └── metrics.py              # Segmentation metrics (mIoU)
├── checkpoints/                 # Model checkpoints
├── train.py                    # Training script
├── predict.py                  # Prediction and visualization script
└── .gitignore                  # Git ignore file
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- NumPy
- PIL (Pillow)
- Matplotlib
- PyYAML

## Getting Started

### 1. Prepare Data

Download the PASCAL VOC 2012 dataset and place it in the `data/` directory:

```bash
# Download VOC 2012 dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Extract to data directory
tar -xf VOCtrainval_11-May-2012.tar -C data/
```

### 2. Train the Model

Run the training script:

```bash
python train.py
```

The model will be trained for 50 epochs by default. Checkpoints will be saved in the `checkpoints/` directory.

### 3. Evaluate the Model

The training script automatically evaluates the model on the validation set after each epoch, computing metrics like mIoU (mean Intersection over Union) and pixel accuracy.

### 4. Make Predictions

Run the prediction script to generate segmentation masks for test images:

```bash
python predict.py
```

Results will be saved in the `output/` directory, including:
- `segmentation_mask.png`: The raw segmentation mask
- `overlay.png`: The mask overlaid on the original image
- `result.png`: A comparison of the original image, mask, and overlay

## Model Architecture

The model uses a Fully Convolutional U-Net architecture, which consists of:

1. **Encoder**: Four convolutional blocks with max pooling for downsampling
2. **Bottleneck**: A central convolutional block
3. **Decoder**: Four transposed convolutional blocks for upsampling
4. **Final Layer**: A 1x1 convolutional layer to map features to class probabilities

## Configuration

The `configs/voc_seg.yaml` file contains all configuration parameters, including:

- Data settings (batch size, image size, etc.)
- Model settings (number of classes, feature channels, etc.)
- Training settings (epochs, learning rate, etc.)
- VOC color palette for segmentation masks

## Metrics

The project uses the following metrics for evaluation:

- **mIoU (mean Intersection over Union)**: The average IoU across all classes
- **Pixel Accuracy**: The percentage of correctly classified pixels

## License

This project is licensed under the MIT License.

## Acknowledgments

- PASCAL VOC 2012 dataset
- PyTorch framework