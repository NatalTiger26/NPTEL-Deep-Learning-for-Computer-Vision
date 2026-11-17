"""Unit tests for Assignment 1: PyTorch tensors, autograd, and MLP architecture."""

import torch
import torch.nn as nn
from tests.conftest import extract_symbols_from_notebook


def test_assignment1_mlp_architecture_and_parameters():
    """Verify MLP network structure and parameter count equals 8,197."""
    symbols = extract_symbols_from_notebook("DL4CV_Assignment_1_2023.ipynb", ["MLP"])
    assert "MLP" in symbols, "MLP class should be defined in DL4CV_Assignment_1_2023.ipynb"
    MLP = symbols["MLP"]

    num_features = 10
    num_classes = 5
    num_hidden = 512

    model = MLP(num_features, num_classes, num_hidden)

    # Check layer configurations
    assert isinstance(model.hidden, nn.Linear)
    assert model.hidden.in_features == 10
    assert model.hidden.out_features == 512

    assert isinstance(model.relu, nn.ReLU)

    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.in_features == 512
    assert model.classifier.out_features == 5

    # Check total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 8197, f"Expected 8197 parameters, got {total_params}"

    # Forward pass shape check
    batch_size = 16
    x = torch.randn(batch_size, num_features)
    out = model(x)
    assert out.shape == (batch_size, num_classes)


def test_assignment1_tensor_operations_q1_q2_q3():
    """Verify mathematical consistency of tensor operations in Assignment 1."""
    # Q1 test
    torch.manual_seed(2)
    x = torch.randint(low=0, high=256, size=(1, 250))
    x_resize = x.view(10, 5, 5)
    y1 = torch.randn(5, 5)
    y2 = torch.rand(5, 5)

    mul_output1 = x_resize * y1
    final_output1 = torch.sum(mul_output1, dim=(-2, -1))
    mul_output2 = x_resize * y2
    final_output2 = torch.sum(mul_output2, dim=(-2, -1))

    mean_sum = torch.mean(final_output1 + final_output2).item()
    assert abs(mean_sum - 2000) < 100, f"Mean sum {mean_sum} should be closest to 2000"

    # Q2 test
    torch.manual_seed(0)
    labels = torch.randint(low=0, high=2000, size=(10, 2)).type(torch.float)
    o1 = final_output1.unsqueeze(1)
    o2 = final_output2.unsqueeze(1)
    final_output = torch.cat([o1, o2], dim=1)
    logits = torch.relu(final_output)
    eucledian_dist = torch.norm(labels - logits, p=2, dim=1)
    loss = torch.sum(eucledian_dist).item()
    assert abs(loss - 12000) < 500, f"Loss {loss} should be closest to 12000"

    # Q3 test: draw in identical RNG sequence as Assignment 1 cell 11
    torch.manual_seed(0)
    labels_q3 = torch.randint(low=-800, high=50, size=(3, 1)).type(torch.float)
    w = torch.randn(3).requires_grad_()
    X = torch.rand(3, 3)
    pred = torch.matmul(X, w)
    temp = torch.norm(labels_q3 - pred, p=1, dim=1)
    loss_q3 = torch.sum(temp)
    loss_q3.backward(retain_graph=True)
    grad = torch.autograd.grad(loss_q3, w)[0]
    expected_grad = torch.tensor([4.4869, 2.1056, 3.4464])
    assert torch.allclose(grad, expected_grad, atol=1e-3)
