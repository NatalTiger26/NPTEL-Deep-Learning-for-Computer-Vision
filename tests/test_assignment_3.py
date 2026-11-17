"""Unit tests for Assignment 3: Metric Learning, Object Detection Loss, and Recurrent Models."""

from typing import Tuple, Union
import numpy as np
import pytest
import torch
import torch.nn as nn


def get_iou(ground_truth: np.ndarray, pred: np.ndarray) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    i_width = np.maximum(ix2 - ix1, 0.0)
    i_height = np.maximum(iy2 - iy1, 0.0)
    inter_area = i_width * i_height

    gt_area = np.maximum(ground_truth[2] - ground_truth[0], 0.0) * np.maximum(ground_truth[3] - ground_truth[1], 0.0)
    pred_area = np.maximum(pred[2] - pred[0], 0.0) * np.maximum(pred[3] - pred[1], 0.0)

    union_area = gt_area + pred_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining classification (Cross-Entropy) and bounding box regression (Smooth L1)."""

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.smooth_l1_loss: nn.SmoothL1Loss = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        true_class: torch.Tensor,
        bounding_box_offsets: torch.Tensor,
        bounding_box_target: torch.Tensor,
    ) -> torch.Tensor:
        p_logits = pred_logits.unsqueeze(0) if pred_logits.dim() == 1 else pred_logits
        t_class = true_class.unsqueeze(0) if true_class.dim() == 1 else true_class
        cls_loss = self.cross_entropy_loss(p_logits, t_class)
        reg_loss = self.smooth_l1_loss(bounding_box_offsets, bounding_box_target)
        return cls_loss + self.alpha * reg_loss


class LSTM(nn.Module):
    """LSTM recurrent neural network for univariate time series forecasting."""

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        last_step = out[:, -1, :]
        return self.fc(last_step)


def sliding_windows(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create input-target sequence pairs for time-series forecasting."""
    n_samples = len(data) - seq_length - 1
    if n_samples <= 0:
        return np.empty((0, seq_length)), np.empty((0,))
    x = np.array([data[i : i + seq_length] for i in range(n_samples)])
    y = np.array([data[i + seq_length] for i in range(n_samples)])
    return x, y


def test_iou_calculation() -> None:
    """Test IoU metric on identical, non-overlapping, and question boxes."""
    # Identical boxes
    box_a = np.array([0, 0, 10, 10], dtype=np.float32)
    assert np.isclose(get_iou(box_a, box_a), 1.0)

    # Disjoint boxes
    box_b = np.array([20, 20, 30, 30], dtype=np.float32)
    assert np.isclose(get_iou(box_a, box_b), 0.0)

    # Question 1 test bounding boxes
    gt_bbox = np.array([1100, 113, 1450, 768], dtype=np.float32)
    pred_bbox = np.array([1262.0001, 102.0021, 1519.9832, 794.0033], dtype=np.float32)
    iou = get_iou(gt_bbox, pred_bbox)
    assert np.isclose(iou, 0.4326, atol=1e-3)


def test_multitask_loss() -> None:
    """Test combined classification and bounding box regression loss."""
    mt_loss = MultiTaskLoss(alpha=0.01)
    predicted_logits = torch.tensor([1.4315, 2.9862, 0.4019, 1.5501])
    true_class = torch.tensor([0.0, 1.0, 0.0, 0.0])
    bounding_box_offsets = torch.tensor([
        [25.0, 51.0, 82.0, 120.0],
        [63.0, 42.0, 88.0, 112.0],
        [29.0, 45.0, 80.0, 117.0],
        [48.0, 39.0, 92.0, 105.0],
    ])
    bounding_box_target = torch.tensor([
        [27.0, 49.0, 84.0, 118.0],
        [60.0, 45.0, 90.0, 110.0],
        [31.0, 43.0, 82.0, 115.0],
        [45.0, 42.0, 90.0, 108.0],
    ])
    loss_val = mt_loss(predicted_logits, true_class, bounding_box_offsets, bounding_box_target).item()
    assert np.isclose(loss_val, 0.7117, atol=1e-3)


def test_lstm_architecture_and_parameters() -> None:
    """Test LSTM parameter count (117) and forward pass shape."""
    num_classes = 1
    input_size = 1
    hidden_size = 4
    num_layers = 1

    model = LSTM(num_classes, input_size, hidden_size, num_layers)

    # Param count: 4*(input_size*hidden_size + hidden_size^2 + 2*hidden_size) + (hidden_size*num_classes + num_classes)
    # = 4*(1*4 + 16 + 8) + (4*1 + 1) = 4*28 + 5 = 112 + 5 = 117
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 117

    x = torch.randn(10, 5, input_size)
    pred = model(x)
    assert pred.shape == (10, 1)


def test_sliding_windows() -> None:
    """Test sliding windows generator."""
    data = np.arange(20, dtype=np.float32)
    seq_len = 5
    x, y = sliding_windows(data, seq_len)
    assert x.shape == (14, 5)
    assert y.shape == (14,)
    assert (x[0] == data[:5]).all()
    assert y[0] == data[5]


def test_gru_step_computation() -> None:
    """Test GRU manual equation forward computation."""
    h_size = 20
    N = 60
    torch.manual_seed(0)

    Wz = torch.rand((h_size, N))
    Wr = torch.rand((h_size, N))
    Wh = torch.rand((h_size, N))

    x = torch.randn((40, 1))
    h_prev = torch.randn((20, 1))

    inp = torch.cat((h_prev, x), dim=0)
    z = torch.sigmoid(torch.matmul(Wz, inp))
    r = torch.sigmoid(torch.matmul(Wr, inp))
    temp = torch.cat((x, r * h_prev), dim=0)
    h_hat = torch.tanh(torch.matmul(Wh, temp))
    h_t = (1.0 - z) * h_prev + z * h_hat

    mean_ht = torch.mean(h_t).item()
    assert np.isclose(mean_ht, -0.3015, atol=1e-3)
