"""Unit tests for Assignment 1: PyTorch Basics, Autograd, and MLP Classification."""

import torch
import torch.nn as nn
from typing import Tuple


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) for multi-class classification."""

    def __init__(self, num_features: int, num_classes: int, num_hidden: int) -> None:
        super(MLP, self).__init__()
        self.hidden: nn.Linear = nn.Linear(num_features, num_hidden)
        self.relu: nn.ReLU = nn.ReLU()
        self.classifier: nn.Linear = nn.Linear(num_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.hidden(x)
        out = self.relu(out)
        out = self.classifier(out)
        return out


def test_tensor_operations() -> None:
    """Test basic tensor rearrangement, random generation, and elementwise multiplication."""
    torch.manual_seed(2)
    x: torch.Tensor = torch.randint(low=0, high=256, size=(1, 250))
    x_resize: torch.Tensor = x.view(10, 5, 5)
    assert x_resize.shape == (10, 5, 5)

    y1: torch.Tensor = torch.randn(5, 5)
    y2: torch.Tensor = torch.rand(5, 5)

    mul_output1: torch.Tensor = x_resize * y1
    final_output1: torch.Tensor = torch.sum(mul_output1, dim=(-2, -1))
    assert final_output1.shape == (10,)

    mul_output2: torch.Tensor = x_resize * y2
    final_output2: torch.Tensor = torch.sum(mul_output2, dim=(-2, -1))
    assert final_output2.shape == (10,)

    mean_sum: float = torch.mean(final_output1 + final_output2).item()
    assert isinstance(mean_sum, float)


def test_l2_loss_and_relu() -> None:
    """Test concatenation, ReLU activation, and Euclidean norm computation."""
    torch.manual_seed(0)
    final_output1 = torch.randn(10)
    final_output2 = torch.randn(10)
    labels = torch.randint(low=0, high=2000, size=(10, 2)).float()

    o1 = final_output1.unsqueeze(1)
    o2 = final_output2.unsqueeze(1)
    final_output = torch.cat([o1, o2], dim=1)
    assert final_output.shape == (10, 2)

    logits = torch.relu(final_output)
    assert (logits >= 0.0).all()

    eucledian_dist = torch.norm(labels - logits, p=2, dim=1)
    assert eucledian_dist.shape == (10,)
    assert (eucledian_dist >= 0.0).all()


def test_autograd_gradients() -> None:
    """Test automatic differentiation on matrix-vector product with L1 norm."""
    torch.manual_seed(0)
    labels = torch.randint(low=-800, high=50, size=(3, 1)).float()
    w = torch.randn(3, requires_grad=True)
    X = torch.rand(3, 3)

    pred = torch.matmul(X, w)
    temp = torch.norm(labels - pred, p=1, dim=1)
    loss = torch.sum(temp)

    grad = torch.autograd.grad(loss, w)[0]
    assert grad.shape == (3,)
    assert not torch.isnan(grad).any()


def test_mlp_parameters_and_forward() -> None:
    """Test MLP parameter count and forward pass shape."""
    num_features = 10
    num_classes = 5
    num_hidden = 512

    model = MLP(num_features, num_classes, num_hidden)
    total_params = sum(p.numel() for p in model.parameters())

    # Hidden: 10 * 512 + 512 = 5632; Classifier: 512 * 5 + 5 = 2565; Total = 8197
    expected_params = (num_features * num_hidden + num_hidden) + (num_hidden * num_classes + num_classes)
    assert total_params == expected_params
    assert total_params == 8197

    batch_x = torch.randn(16, num_features)
    logits = model(batch_x)
    assert logits.shape == (16, num_classes)
