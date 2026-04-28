# Advanced Deep Learning Architectures

Implementing state-of-the-art deep learning models from scratch. Exploring the evolution of Computer Vision architectures using PyTorch and TensorFlow/Keras.

## Contents

### 01 - ResNet (Residual Networks)
- **Status:** Completed
- **Architecture:** ResNet50 (Built from scratch using Keras Functional API)
- **Key Components:** Implemented custom Identity Blocks and Convolutional Blocks with skip connections to solve the vanishing gradient problem.
- **Optimization & Hardware:** - Utilized `mixed_float16` precision to successfully train the massive architecture locally on a 4GB RTX 3050.
  - Implemented Gradient Clipping (`clipnorm=1.0`) to prevent exploding gradients.
  - Used `EarlyStopping` and `ReduceLROnPlateau` callbacks to capture optimal weights and stabilize the learning rate.
- **Performance:**
  - Training Accuracy: 100%
  - Validation Accuracy: 95.37%
  - Test Accuracy: 92.50%

### 02 - Inception Networks
- *Status: Planned*
- Implementing Inception modules with 1x1 convolutions for dimensionality reduction.

### 03 - MobileNet
- *Status: Planned*
- Implementing Depthwise Separable Convolutions for lightweight, efficient mobile models.

## Environment Setup

```bash
git clone git@github.com:dikshantbisht1308/advanced-dl-architectures.git
cd advanced-dl-architectures
conda create -n advanced-dl python=3.10 -y
conda activate advanced-dl
pip install -r requirements.txt

Stack
Python · PyTorch · TensorFlow · Keras · NumPy · Matplotlib · Jupyter

Progress
[ ] ResNet (Residual Networks)

[ ] Inception Network (GoogLeNet)

[ ] MobileNet (v1/v2)

[ ] EfficientNet
