"""Unit tests for Assignment 2: CNN on MNIST, Saliency Maps, and Class Activation Mapping (CAM)."""

from typing import List, Tuple
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn


class ConvolutionNet(nn.Module):
    """Two-layer Convolutional Neural Network with Batch Normalization and Max-Pooling."""

    def __init__(self, num_classes: int = 10) -> None:
        super(ConvolutionNet, self).__init__()
        self.conv_block1: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block2: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc: nn.Linear = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.conv_block1(x)
        output = self.conv_block2(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def CAM(
    conv_feature: torch.Tensor,
    wts: torch.Tensor,
    cls_list: List[int],
    up_shape: Tuple[int, int] = (224, 224),
) -> List[np.ndarray]:
    """Compute Class Activation Maps (CAM) for specified target classes."""
    batch_size, channels, h, w = conv_feature.shape
    output_cam: List[np.ndarray] = []
    resize_feature: torch.Tensor = torch.reshape(conv_feature, (batch_size, channels, h * w))

    for idx in cls_list:
        cam: torch.Tensor = torch.matmul(wts[idx], resize_feature[0])
        cam = torch.reshape(cam, (h, w))
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max > cam_min:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = torch.zeros_like(cam)

        cam_img: np.ndarray = np.uint8(255 * cam_norm.detach().cpu().numpy())
        output_cam.append(cv2.resize(cam_img, up_shape))

    return output_cam


def test_convolution_net_architecture() -> None:
    """Test ConvolutionNet layer configurations, weights, and parameter count."""
    model = ConvolutionNet(num_classes=10)

    # Check conv_block2 weight shape: [16, 32, 7, 7]
    assert model.conv_block2[0].weight.shape == torch.Size([16, 32, 7, 7])

    # Parameter count calculation:
    # Block 1: Conv2d(1, 32, 5): 32*1*25 + 32 = 832; BatchNorm2d(32): 64 (weight + bias)
    # Block 2: Conv2d(32, 16, 7): 16*32*49 + 16 = 25104; BatchNorm2d(16): 32
    # FC: Linear(16*7*7=784, 10): 784 * 10 + 10 = 7850
    # Total: 832 + 64 + 25104 + 32 + 7850 = 33882
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 33882

    # Forward pass shape check
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_vectorized_saliency() -> None:
    """Verify that vectorized spatial channel maximums match iterative loop exactly."""
    torch.manual_seed(42)
    absolute_value = torch.randn(3, 224, 224).abs()

    # Iterative method
    max_values = []
    for i in range(absolute_value.shape[0]):
        max_values.append(absolute_value[i].max())
    loop_mean = (sum(max_values) / len(max_values)).item()

    # Vectorized method
    vectorized_mean = absolute_value.amax(dim=(-2, -1)).mean().item()

    assert np.isclose(loop_mean, vectorized_mean, atol=1e-6)


def test_cam_computation() -> None:
    """Test Class Activation Map (CAM) generation."""
    torch.manual_seed(2022)
    conv_feature = torch.randn((1, 512, 7, 7))
    wts = torch.rand((10, 512))

    cam_maps = CAM(conv_feature, wts, [6], up_shape=(224, 224))
    assert len(cam_maps) == 1
    assert cam_maps[0].shape == (224, 224)
    assert cam_maps[0].dtype == np.uint8
    assert cam_maps[0].min() >= 0
    assert cam_maps[0].max() <= 255


def test_filter_median_calculation() -> None:
    """Test Question 6 median filter combination."""
    a = np.array([120, 5, 100, 60, 40, 80, 30, 30, 20])
    b = np.array([50, 100, 50, 70, 110, 60, 140, 130, 170])
    c = np.array([90, 75, 45, 200, 110, 55, 10, 70, 80])
    combined = 0.1 * a + 0.2 * b + 0.3 * c
    med = float(np.median(combined))
    assert isinstance(med, float)
    assert med > 0.0
