
# NeRF Primer

This repository contains a minimal implementation of a Neural Radiance Fields (NeRF) model using PyTorch. The code includes generating synthetic data, defining the NeRF model, training, and rendering images.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Bcw1q_JbRx4eCDhCiOUcn8f_2r3_Jm-G?usp=sharinghttps://colab.research.google.com/drive/1Bcw1q_JbRx4eCDhCiOUcn8f_2r3_Jm-G?usp=sharing)

## Repository Structure

- `dataset.py`: Contains the `DummyCubeDataset` class for generating a dataset of synthetic cube images and their corresponding camera poses.
- `model.py`: Defines the NeRF model architecture and the positional encoding function.
- `train_utils.py`: Includes utility functions for training the NeRF model, such as ray generation, sampling points, and rendering images.
- `train_encoding.py`: Script for training the NeRF model with positional encoding.
- `train_no_encoding.py`: Script for training the NeRF model without positional encoding.

## Modules

### `dataset.py`

This module defines the `DummyCubeDataset` class, which generates synthetic images of a cube from different camera poses.

#### How to Use
```python
from dataset import DummyCubeDataset
from torch.utils.data import DataLoader

# Initialize dataset
dataset = DummyCubeDataset(num_images=100, H=1000, W=1000, focal=1500, output_dir="output_images")

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### `model.py`

This module defines the NeRF model architecture using PyTorch. It includes the positional encoding function and the NeRF class.

#### How to Use
```python
from model import NeRF

# Initialize model
model = NeRF(D=8, W=256, input_ch=6, output_ch=4, L=10)
```

### `train_utils.py`

This module contains various utility functions used for training the NeRF model, such as generating rays, sampling points along rays, rendering rays, and training the model.

#### Key Functions
- `seed_everything(seed)`: Seeds all random number generators for reproducibility.
- `rotation_matrix(elevation, azimuth)`: Generates a rotation matrix.
- `pose_to_matrix(pose)`: Converts pose parameters to a 4x4 transformation matrix.
- `get_rays(H, W, focal, c2w)`: Generates rays from camera poses.
- `sample_points_along_rays(rays_o, rays_d, num_samples, near, far)`: Samples points along the generated rays.
- `render_rays(model, rays_o, rays_d, num_samples, near, far)`: Renders the colors along the rays using the NeRF model.
- `generate_rays_and_rgb(images, poses, H, W, focal)`: Generates rays and RGB values from images and poses.
- `train_nerf(model, dataloader, epochs, lr, H, W, focal, num_samples, near, far)`: Trains the NeRF model.
- `inference_nerf(model, H, W, focal, epoch, num_samples, near, far)`: Performs inference using the trained NeRF model.
- `render_image(model, pose, H, W, focal, num_samples, near, far)`: Renders an image from a given pose.
- `visualize(model, H, W, focal, num_samples, near, far, inference_folder, azimuth_list)`: Generates a video visualization of the rendered images.

### `train_encoding.py`

This script trains the NeRF model with positional encoding. It sets up the dataset, initializes the model, and runs the training loop.

#### How to Use
```bash
python train_encoding.py
```

### `train_no_encoding.py`

This script trains the NeRF model without positional encoding. It sets up the dataset, initializes the model, and runs the training loop.

#### How to Use
```bash
python train_no_encoding.py
```

## Getting Started

1. **Clone the repository**:
```bash
git clone https://github.com/your_username/nerf_primer.git
cd nerf_primer
```

2. **Install dependencies**:
Make sure you have PyTorch installed. You can install other dependencies using pip:
```bash
pip install -r requirements.txt
```

3. **Run training**:
You can start training the NeRF model with positional encoding:
```bash
python train_encoding.py
```
Or without positional encoding:
```bash
python train_no_encoding.py
```

## Results

The trained models will generate images and save them in the specified directories. You can visualize the training progress in terminal. It also generate a video visualization of the rendered images at the end of the training that you can find in the `inference_folder` directory.

## Acknowledgements

This implementation is based on the paper [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
