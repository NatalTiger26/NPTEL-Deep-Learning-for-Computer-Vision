"""Unit tests for Assignment 2: ConvolutionNet, Saliency, Occlusion, and CAM."""

import numpy as np
import torch
import torch.nn as nn
from tests.conftest import extract_symbols_from_notebook


def test_assignment2_convolution_net():
    """Verify ConvolutionNet architecture, channel dimensions, and forward pass."""
    symbols = extract_symbols_from_notebook(
        "drive-download-20230903T003734Z-001/DL4CV_Assignment_2_2023.ipynb",
        ["ConvolutionNet"],
    )
    assert "ConvolutionNet" in symbols
    ConvolutionNet = symbols["ConvolutionNet"]

    model = ConvolutionNet(num_classes=10)
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    out = model(x)
    assert out.shape == (batch_size, 10)

    # Inspect first conv layer shape: (32, 1, 5, 5)
    assert model.conv_block1[0].weight.shape == (32, 1, 5, 5)
    # Inspect second conv layer shape: (16, 32, 7, 7)
    assert model.conv_block2[0].weight.shape == (16, 32, 7, 7)
    # Inspect fc layer shape: (10, 784)
    assert model.fc.weight.shape == (10, 16 * 7 * 7)


def test_assignment2_preprocess_image():
    """Verify image preprocessing normalizes and attaches gradients."""
    symbols = extract_symbols_from_notebook(
        "drive-download-20230903T003734Z-001/DL4CV_Assignment_2_2023.ipynb",
        ["preprocess_image"],
    )
    assert "preprocess_image" in symbols
    preprocess_image = symbols["preprocess_image"]

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = preprocess_image(dummy_img, resize_im=True)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.requires_grad is True


def test_assignment2_cam_multiple_classes():
    """Verify CAM generates heatmaps for multiple classes without losing early classes."""
    symbols = extract_symbols_from_notebook(
        "drive-download-20230903T003734Z-001/DL4CV_Assignment_2_2023.ipynb",
        ["CAM"],
    )
    assert "CAM" in symbols
    CAM = symbols["CAM"]

    torch.manual_seed(2022)
    conv_feature = torch.randn((1, 512, 7, 7))
    wts = torch.rand((10, 512))

    cls_list = [2, 6]
    heatmaps = CAM(conv_feature, wts, cls_list)

    assert len(heatmaps) == 2, f"Expected 2 activation maps, got {len(heatmaps)}"
    for hm in heatmaps:
        assert isinstance(hm, np.ndarray)
        assert hm.shape == (224, 224)
        assert hm.min() >= 0 and hm.max() <= 255
