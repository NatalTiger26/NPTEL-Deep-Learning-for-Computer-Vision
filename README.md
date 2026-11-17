# Deep Learning for Computer Vision (DL4CV)

A modernized, thoroughly documented, and strictly typed repository implementing core and advanced deep learning paradigms for computer vision. This repository contains complete implementations across five progressive assignments covering foundational tensor autograd, multi-layer perceptrons, convolutional architectures, model interpretability, metric learning, object detection objectives, sequential modeling, attention mechanisms, vision transformers, generative models (VAEs, AAEs, CycleGAN), adversarial robustness, and self-supervised learning.

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Curriculum & Methodological Breakdown](#curriculum--methodological-breakdown)
  - [Assignment 1: PyTorch Foundations & Multi-Layer Perceptrons](#assignment-1-pytorch-foundations--multi-layer-perceptrons)
  - [Assignment 2: Convolutional Networks & Visual Interpretability](#assignment-2-convolutional-networks--visual-interpretability)
  - [Assignment 3: Metric Learning, Multi-Task Losses & Recurrent Models](#assignment-3-metric-learning-multi-task-losses--recurrent-models)
  - [Assignment 4: Attention Mechanisms, Vision Transformers & Generative Models](#assignment-4-attention-mechanisms-vision-transformers--generative-models)
  - [Assignment 5: Cycle-Consistent GANs, Adversarial Robustness & SSL](#assignment-5-cycle-consistent-gans-adversarial-robustness--ssl)
- [Environment Setup](#environment-setup)
- [Running the Code & Tests](#running-the-code--tests)
- [Code Quality & Modernization Standards](#code-quality--modernization-standards)

---

## Overview

Modern deep learning for computer vision requires a deep understanding of tensor graph mechanics, spatial representations, feature attribution, latent generative modeling, and self-supervised representations. This codebase modernizes educational reference implementations with:
- **Strict In-Place Typing**: Full static type hints conforming to [PEP 484](https://peps.python.org/pep-0484/) across all functions and classes.
- **Comprehensive Documentation**: Google/NumPy-style docstrings documenting architectures, tensor shapes, inputs, and mathematical formulations.
- **Bug Fixes & Device Agnosticism**: Elimination of deprecated APIs (e.g. `Variable`), silent bugs (e.g. CAM class-loop state overwrites), and automatic fallback across CUDA, Apple Silicon (`mps`), and CPU.
- **Automated Test Suite**: Pytest test suite covering notebook syntax integrity and mathematical guarantees of all implemented models and algorithms.

---

## Repository Structure

All existing notebooks, sample assets, and raw datasets remain exactly in their original relative locations:

```text
.
├── DL4CV_Assignment_1_2023.ipynb               # Assignment 1: PyTorch basics, autograd, MLP
├── DL4CV_Assignment_3_2023.ipynb               # Assignment 3: Metric learning, IoU, LSTM
├── DL4CV_Assignment_5_2023.ipynb               # Assignment 5: CycleGAN, FGSM, SSL rotations
├── drive-download-20230903T003734Z-001/
│   ├── DL4CV_Assignment_2_2023.ipynb           # Assignment 2: CNNs, Saliency, Occlusion, CAM
│   ├── Copy_of_DL4CV_Assignment_2_2023[1].ipynb# Assignment 2 duplicate snapshot
│   ├── car.png                                 # Reference sample image for attribution
│   └── fox.jpg                                 # Reference sample image for occlusion
├── drive-download-20231003T210250Z-001/
│   ├── DL4CV_Assignment_4_2023.ipynb           # Assignment 4: Attention, ViT patch, VAE, AAE
│   ├── car.png                                 # Reference sample image for ViT patching
│   └── MNIST_data/                             # Raw MNIST ubyte dataset files
├── tests/                                      # Automated unit test suite
│   ├── __init__.py
│   ├── conftest.py                             # Test configuration & symbol extraction
│   ├── test_notebooks_integrity.py             # Schema & AST syntax compilation tests
│   ├── test_assignment1_mlp.py                 # Tests for MLP & tensor autograd
│   ├── test_assignment2_cnn_interpretability.py# Tests for CNN & interpretability
│   ├── test_assignment3_metric_and_recurrent.py# Tests for Triplet loss, IoU, LSTM
│   ├── test_assignment4_attention_and_generative.py # Tests for MHA, ViT, KL divergence
│   └── test_assignment5_cyclegan_robustness_ssl.py  # Tests for CycleGAN, FGSM, SSL
├── requirements.txt                            # Dependencies
├── pytest.ini                                  # Pytest configuration
├── .gitignore                                  # Git exclusion configuration
└── README.md                                   # Comprehensive documentation
```

---

## Curriculum & Methodological Breakdown

### Assignment 1: PyTorch Foundations & Multi-Layer Perceptrons
**File:** [DL4CV_Assignment_1_2023.ipynb](file:///Users/nataltiger/github-cleanup/NPTEL-Deep-Learning-for-Computer-Vision/DL4CV_Assignment_1_2023.ipynb)

- **Tensor Manipulations & Broadcasting**:
  - Reshaping 1D random integers $(1, 250)$ into 3D tensors $(10, 5, 5)$.
  - Multi-dimensional broadcasting with Gaussian $\mathcal{N}(0, 1)$ and Uniform $\mathcal{U}[0, 1]$ tensors.
  - Spatial summation across trailing dimensions $\text{dim}=(-2, -1)$.
- **Objective Formulations & Autograd**:
  - Euclidean norm ($L_2$ distance) computation against multi-channel targets.
  - Explicit Jacobian-vector gradient backpropagation using `torch.autograd.grad`.
- **Feedforward Classifier**:
  - `MLP`: Architecture comprising `nn.Linear(10, 512)` $\rightarrow$ `nn.ReLU()` $\rightarrow$ `nn.Linear(512, 5)`.
  - Exact parameter count: $(10 \times 512 + 512) + (512 \times 5 + 5) = 5632 + 2565 = 8197$.
  - End-to-end deterministic training loop with Adam optimizer and `CrossEntropyLoss`.

---

### Assignment 2: Convolutional Networks & Visual Interpretability
**File:** [drive-download-20230903T003734Z-001/DL4CV_Assignment_2_2023.ipynb](file:///Users/nataltiger/github-cleanup/NPTEL-Deep-Learning-for-Computer-Vision/drive-download-20230903T003734Z-001/DL4CV_Assignment_2_2023.ipynb)

- **Convolutional Network Architecture**:
  - `ConvolutionNet`: Two-stage feature extractor with Conv2d, Batch Normalization, ReLU non-linearity, and Max Pooling.
  - Dimension progression: $(B, 1, 28, 28) \rightarrow (B, 32, 14, 14) \rightarrow (B, 16, 7, 7) \rightarrow (B, 10)$.
- **Transfer Learning**:
  - Fine-tuning ImageNet-pretrained ResNet-18 on CIFAR-10 with selective parameter freezing on earlier residual stages (`layer1`, `layer2`).
- **Interpretability & Explainability**:
  - **Vanilla Gradient Saliency**: Computing the gradient of the top-predicted class score with respect to input pixels: $S(x) = \left|\nabla_x S_c(x)\right|$.
  - **Occlusion Sensitivity Analysis**: Measuring bottleneck feature activation divergence when critical input regions are systematically occluded.
  - **Class Activation Mapping (CAM)**: Projecting fully-connected classification weights $w_k^c$ back onto the final convolutional activation maps:
    $$M_c(x, y) = \sum_k w_k^c f_k(x, y)$$
    Upsampled to original image resolution with min-max normalization.

---

### Assignment 3: Metric Learning, Multi-Task Losses & Recurrent Models
**File:** [DL4CV_Assignment_3_2023.ipynb](file:///Users/nataltiger/github-cleanup/NPTEL-Deep-Learning-for-Computer-Vision/DL4CV_Assignment_3_2023.ipynb)

- **Metric Learning with Triplet Loss**:
  - `TripletDataset`: Dynamic sample mining extracting positive pairs $(a, p)$ of matching classes and negative pairs $(a, n)$ of disjoint classes.
  - `EmbeddingNet` & `TripletNet`: CNN mapping image inputs into a 2D Euclidean embedding space.
  - `TripletLoss`: Margin ranking loss:
    $$\mathcal{L}(a, p, n) = \max\left(0, \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha\right)$$
- **Object Detection & Multi-Task Objectives**:
  - `get_iou`: Intersection over Union computation between ground truth and predicted bounding boxes $[x_1, y_1, x_2, y_2]$.
  - `MultiTaskLoss`: Joint objective combining cross-entropy classification and Smooth $L_1$ bounding box regression:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}}(p, u) + \alpha [u \ge 1] \mathcal{L}_{\text{loc}}(t^u, v)$$
- **Recurrent Time-Series Forecasting**:
  - `sliding_windows`: Temporal history window extraction.
  - `LSTM`: Recurrent architecture for sequential regression on harmonic wave data.
  - Numerical implementations of gated activations (`softmax`, `sigmoid`).

---

### Assignment 4: Attention Mechanisms, Vision Transformers & Generative Models
**File:** [drive-download-20231003T210250Z-001/DL4CV_Assignment_4_2023.ipynb](file:///Users/nataltiger/github-cleanup/NPTEL-Deep-Learning-for-Computer-Vision/drive-download-20231003T210250Z-001/DL4CV_Assignment_4_2023.ipynb)

- **Multi-Head Attention**:
  - `MultiHeadAttention`: Implementation of Scaled Dot-Product Multi-Head Attention:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  - Multi-head tensor partitioning, transposition, scaling, and output projection.
- **Vision Transformer (ViT) Tokenization**:
  - `img_patch`: Spatial decomposition of image $(B, C, H, W)$ into non-overlapping $(P \times P)$ patches and flattening into 1D visual tokens $(B, N, P^2 C)$.
- **Latent Divergence & Generative Autoencoders**:
  - **Analytical KL Divergence**:
    $$D_{\text{KL}}(q(z|x) \parallel \mathcal{N}(0, I)) = -\frac{1}{2} \sum \left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$
  - **Monte Carlo KL Divergence**: Expectation estimation via reparameterized samples:
    $$D_{\text{KL}}^{\text{MC}} = \mathbb{E}_q \left[\log q(z|x) - \log p(z)\right]$$
  - **Adversarial Autoencoders (AAE)**: Encoder $q(z|x)$, Decoder $p(x|z)$, and Discriminator $D(z)$ aligning the aggregated posterior to an arbitrary Gaussian prior.

---

### Assignment 5: Cycle-Consistent GANs, Adversarial Robustness & SSL
**File:** [DL4CV_Assignment_5_2023.ipynb](file:///Users/nataltiger/github-cleanup/NPTEL-Deep-Learning-for-Computer-Vision/DL4CV_Assignment_5_2023.ipynb)

- **CycleGAN**:
  - Dual generators $G_{A \to B}, G_{B \to A}$ and discriminators $D_A, D_B$.
  - Adversarial BCE loss coupled with cycle consistency loss:
    $$\mathcal{L}_{\text{cyc}} = \mathbb{E}[\|G_{B \to A}(G_{A \to B}(x_A)) - x_A\|_1] + \mathbb{E}[\|G_{A \to B}(G_{B \to A}(x_B)) - x_B\|_1]$$
- **Perceptual / Content Loss**:
  - Deep feature extraction using intermediate VGG16 representations to compute semantic content distance $\text{MSE}(\phi(x), \phi(y))$.
- **Adversarial Robustness (FGSM)**:
  - Fast Gradient Sign Method generating bounded $L_\infty$ adversarial perturbations:
    $$x_{\text{adv}} = \text{clip}\left(x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(\theta, x, y)\right), 0, 1\right)$$
- **Self-Supervised Learning (SSL)**:
  - Pretext task predicting 4-fold image rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
  - `Net`: Dual-headed LeNet with joint primary classification (`fc2`, 10 classes) and self-supervised rotation prediction (`fc2_ssl`, 4 classes).
  - Contrastive normalized similarity estimation (InfoNCE / NT-Xent loss).

---

## Environment Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ & Torchvision

### Installation via uv or pip
```bash
# Clone the repository
git clone https://github.com/NatalTiger26/NPTEL-Deep-Learning-for-Computer-Vision.git
cd NPTEL-Deep-Learning-for-Computer-Vision

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Code & Tests

### Executing Notebooks
Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab
```
Open any of the assignment notebooks. Code cells execute deterministically with fixed random seeds.

### Running Automated Tests
The repository includes a comprehensive pytest suite verifying syntax integrity and mathematical guarantees:
```bash
pytest -v
```
All 28 tests validate:
- Valid JSON format and syntax compilation across all 6 notebooks.
- Exact parameter counts and tensor shapes for `MLP`, `ConvolutionNet`, `LeNet`, and `Net`.
- Mathematical equivalence and stability of `get_iou`, `MultiTaskLoss`, `TripletLoss`, `img_patch`, `KL_analytical`, `kl_mc`, and `fgsm_attack`.

---

## Code Quality & Modernization Standards

All code adheres strictly to the following standards:
1. **PEP 484 Type Hints**: All functions, methods, and classes feature static typing (`torch.Tensor`, `np.ndarray`, `Tuple`, `Optional`, `Callable`, `Dict`, `List`).
2. **Google/NumPy Docstrings**: Comprehensive docstrings detailing module architecture, dimension progression, argument types, and mathematical formulas.
3. **Device Agnostic Execution**: No hardcoded device assignments. Automatically detects and leverages CUDA, Apple Silicon (`mps`), or CPU without runtime crashes.
4. **Clean Code Hygiene**: Zero dead code, no unused imports (e.g. removed deprecated `Variable` imports), and zero broken package installation scripts.
