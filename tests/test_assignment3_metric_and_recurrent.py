"""Unit tests for Assignment 3: Metric learning, IoU, Multi-task loss, LSTM, and activations."""

import numpy as np
import torch
from tests.conftest import extract_symbols_from_notebook


def test_assignment3_get_iou():
    """Verify IoU calculation on identical, non-overlapping, and realistic bounding boxes."""
    symbols = extract_symbols_from_notebook("DL4CV_Assignment_3_2023.ipynb", ["get_iou"])
    assert "get_iou" in symbols
    get_iou = symbols["get_iou"]

    # Identical boxes
    box_a = np.array([10.0, 10.0, 50.0, 50.0])
    assert abs(get_iou(box_a, box_a) - 1.0) < 1e-4

    # Disjoint boxes
    box_b = np.array([100.0, 100.0, 150.0, 150.0])
    assert get_iou(box_a, box_b) == 0.0

    # Realistic assignment test case
    gt_bbox = np.array([1100.0, 113.0, 1450.0, 768.0], dtype=np.float32)
    pd_bbox = np.array([1262.0001, 102.0021, 1519.9832, 794.0033], dtype=np.float32)
    iou = get_iou(gt_bbox, pd_bbox)
    assert abs(iou - 0.43389) < 1e-3, f"Expected IoU ~0.43389, got {iou}"


def test_assignment3_multi_task_loss():
    """Verify MultiTaskLoss handles background (class 0) and foreground classes correctly."""
    symbols = extract_symbols_from_notebook("DL4CV_Assignment_3_2023.ipynb", ["MultiTaskLoss"])
    assert "MultiTaskLoss" in symbols
    MultiTaskLoss = symbols["MultiTaskLoss"]

    mt_loss = MultiTaskLoss(alpha=0.1)

    # Class 0: background -> Localization loss should be 0
    logits = torch.tensor([5.0, 1.0, 0.5])
    true_class_bg = torch.tensor([1.0, 0.0, 0.0])
    offsets = torch.randn(3, 4)
    target = torch.randn(4)

    loss_bg = mt_loss(logits, true_class_bg, offsets, target)
    expected_cls = -torch.log(torch.softmax(logits, dim=-1)[0])
    assert torch.allclose(loss_bg, expected_cls, atol=1e-5)

    # Class 1: foreground -> Localization loss is included and scaled by alpha
    true_class_fg = torch.tensor([0.0, 1.0, 0.0])
    loss_fg = mt_loss(logits, true_class_fg, offsets, target)
    expected_cls_fg = -torch.log(torch.softmax(logits, dim=-1)[1])
    assert loss_fg > expected_cls_fg


def test_assignment3_triplet_net_and_loss():
    """Verify EmbeddingNet, TripletNet, and TripletLoss margin properties."""
    symbols = extract_symbols_from_notebook(
        "DL4CV_Assignment_3_2023.ipynb",
        ["EmbeddingNet", "TripletNet", "TripletLoss"],
    )
    assert "EmbeddingNet" in symbols and "TripletNet" in symbols and "TripletLoss" in symbols
    EmbeddingNet = symbols["EmbeddingNet"]
    TripletNet = symbols["TripletNet"]
    TripletLoss = symbols["TripletLoss"]

    emb = EmbeddingNet()
    trip_net = TripletNet(emb)
    loss_fn = TripletLoss(margin=1.0)

    x1 = torch.randn(2, 1, 28, 28)
    x2 = torch.randn(2, 1, 28, 28)
    x3 = torch.randn(2, 1, 28, 28)

    e1, e2, e3 = trip_net(x1, x2, x3)
    assert e1.shape == (2, 2)
    assert e2.shape == (2, 2)
    assert e3.shape == (2, 2)

    loss = loss_fn(e1, e2, e3)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_assignment3_lstm_and_sliding_windows():
    """Verify LSTM sequential model and sliding_windows utility."""
    symbols = extract_symbols_from_notebook(
        "DL4CV_Assignment_3_2023.ipynb",
        ["LSTM", "sliding_windows", "softmax", "sigmoid"],
    )
    assert "LSTM" in symbols and "sliding_windows" in symbols
    LSTM = symbols["LSTM"]
    sliding_windows = symbols["sliding_windows"]
    softmax = symbols["softmax"]
    sigmoid = symbols["sigmoid"]

    # Test sliding windows
    raw_data = np.arange(20, dtype=float)
    seq_len = 5
    X, y = sliding_windows(raw_data, seq_len)
    assert X.shape == (14, 5)
    assert y.shape == (14,)
    assert np.allclose(X[0], [0, 1, 2, 3, 4])
    assert y[0] == 5

    # Test LSTM model
    lstm_model = LSTM(num_classes=1, input_size=1, hidden_size=4, num_layers=1)
    inp = torch.randn(8, seq_len, 1)
    out = lstm_model(inp)
    assert out.shape == (8, 1)

    # Test softmax & sigmoid
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(np.sum(probs) - 1.0) < 1e-6
    sig_val = sigmoid(0.0)
    assert abs(sig_val - 0.5) < 1e-6
