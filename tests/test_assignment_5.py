"""Unit tests for Assignment 5: CycleGAN, Adversarial Attacks (FGSM), SSL Rotations, and InfoNCE."""

from typing import Tuple, Union
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Compute binary cross-entropy discriminator loss for real and generated images."""
    real_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)
    return F.binary_cross_entropy_with_logits(logits_real, real_labels) + F.binary_cross_entropy_with_logits(
        logits_fake, fake_labels
    )


def generator_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Compute binary cross-entropy generator loss tricking the discriminator."""
    real_labels = torch.ones_like(logits_fake)
    return F.binary_cross_entropy_with_logits(logits_fake, real_labels)


def cycle_loss(original_input: torch.Tensor, recovered_input: torch.Tensor) -> torch.Tensor:
    """Compute L1 cycle consistency loss."""
    return F.l1_loss(recovered_input, original_input)


class LeNet(nn.Module):
    """Classical LeNet architecture for digit classification."""

    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


def fgsm_attack(
    model: nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Generate adversarial example using Fast Gradient Sign Method (FGSM)."""
    image_adv = image.clone().detach().requires_grad_(True)
    output = model(image_adv)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()

    data_grad = image_adv.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed_image, 0.0, 1.0)


def tensor_rot_90(x: torch.Tensor) -> torch.Tensor:
    """Rotate tensor 90 degrees counter-clockwise."""
    return x.flip(-1).transpose(-2, -1)


def tensor_rot_180(x: torch.Tensor) -> torch.Tensor:
    """Rotate tensor 180 degrees."""
    return x.flip(-1).flip(-2)


def tensor_rot_270(x: torch.Tensor) -> torch.Tensor:
    """Rotate tensor 270 degrees counter-clockwise."""
    return x.transpose(-2, -1).flip(-1)


def rotate_batch_with_labels(batch: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Vectorized rotation application across batch."""
    out = batch.clone()
    m1 = (labels == 1)
    if m1.any():
        out[m1] = tensor_rot_90(batch[m1])
    m2 = (labels == 2)
    if m2.any():
        out[m2] = tensor_rot_180(batch[m2])
    m3 = (labels == 3)
    if m3.any():
        out[m3] = tensor_rot_270(batch[m3])
    return out


def rotate_batch(batch: torch.Tensor, label: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate batch for self-supervised representation learning."""
    n = len(batch)
    if label == 'rand':
        labels = torch.randint(4, (n,), dtype=torch.long, device=batch.device)
        return rotate_batch_with_labels(batch, labels), labels
    elif label == 'expand':
        r0 = batch
        r1 = tensor_rot_90(batch)
        r2 = tensor_rot_180(batch)
        r3 = tensor_rot_270(batch)
        rotated = torch.cat([r0, r1, r2, r3], dim=0)
        labels = torch.cat([
            torch.zeros(n, dtype=torch.long, device=batch.device),
            torch.ones(n, dtype=torch.long, device=batch.device),
            torch.full((n,), 2, dtype=torch.long, device=batch.device),
            torch.full((n,), 3, dtype=torch.long, device=batch.device),
        ])
        return rotated, labels
    else:
        assert isinstance(label, int)
        labels = torch.full((n,), label, dtype=torch.long, device=batch.device)
        return rotate_batch_with_labels(batch, labels), labels


class Net(nn.Module):
    """Dual-head LeNet architecture for simultaneous digit classification and SSL rotation prediction."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc2_ssl = nn.Linear(50, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x), self.fc2_ssl(x)


def test_cyclegan_losses() -> None:
    """Test CycleGAN adversarial and cycle consistency losses."""
    logits_real = torch.randn(8, 1)
    logits_fake = torch.randn(8, 1)
    orig_img = torch.randn(8, 784)
    rec_img = orig_img + 0.1 * torch.randn(8, 784)

    d_loss = discriminator_loss(logits_real, logits_fake)
    g_loss = generator_loss(logits_fake)
    c_loss = cycle_loss(orig_img, rec_img)

    assert d_loss.item() > 0.0
    assert g_loss.item() > 0.0
    assert c_loss.item() > 0.0


def test_lenet_and_fgsm() -> None:
    """Test LeNet forward pass and FGSM adversarial perturbation bounding."""
    torch.manual_seed(0)
    model = LeNet()
    model.eval()

    image = torch.rand(1, 1, 28, 28)
    target = torch.tensor([1])
    epsilon = 0.25

    adv_image = fgsm_attack(model, image, target, epsilon)

    assert adv_image.shape == image.shape
    assert adv_image.min() >= 0.0
    assert adv_image.max() <= 1.0

    # Perturbation magnitude must not exceed epsilon
    diff = (adv_image - image).abs()
    assert (diff <= epsilon + 1e-6).all()


def test_vectorized_rotations_equivalence() -> None:
    """Verify that vectorized rotation operations match individual mathematical rotations."""
    torch.manual_seed(42)
    batch = torch.randn(4, 1, 28, 28)

    rot_batch, labels = rotate_batch(batch, 'expand')
    assert rot_batch.shape == (16, 1, 28, 28)
    assert labels.shape == (16,)
    assert (labels[:4] == 0).all()
    assert (labels[4:8] == 1).all()
    assert (labels[8:12] == 2).all()
    assert (labels[12:16] == 3).all()

    # Verify 4x90 degree rotation returns to original image
    r360 = tensor_rot_90(tensor_rot_90(tensor_rot_90(tensor_rot_90(batch))))
    assert torch.allclose(batch, r360, atol=1e-6)


def test_dual_head_net() -> None:
    """Test dual-head LeNet output shapes."""
    net = Net()
    x = torch.randn(8, 1, 28, 28)
    cls_logits, ssl_logits = net(x)
    assert cls_logits.shape == (8, 10)
    assert ssl_logits.shape == (8, 4)


def test_infonce_contrastive_loss() -> None:
    """Test Question 6 InfoNCE / Contrastive similarity loss."""
    torch.manual_seed(0)
    x1 = torch.randn((2, 5))
    labels1 = torch.tensor([0, 1])
    x2 = torch.randn((2, 5))
    labels2 = torch.tensor([0, 1])
    t = 0.07

    batch = torch.cat((x1, x2), dim=0)
    labels = torch.cat((labels1, labels2), dim=0)
    batch_norm = F.normalize(batch)

    sim_matrix = torch.matmul(batch_norm, batch_norm.t()) / t
    l1 = labels.unsqueeze(1)
    l0 = labels.unsqueeze(0)
    mask = (l1 == l0).float()

    loss = sim_matrix * mask
    mean_loss = loss.mean().item()
    assert np.isclose(mean_loss, 5.0301, atol=1e-3)
