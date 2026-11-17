"""Unit tests for Assignment 5: CycleGAN losses, LeNet, FGSM adversarial attack, and SSL rotations."""

import torch
import torch.nn as nn
from tests.conftest import extract_symbols_from_notebook


def test_assignment5_lenet_and_fgsm():
    """Verify LeNet classification and Fast Gradient Sign Method (FGSM) adversarial attack."""
    symbols = extract_symbols_from_notebook(
        "DL4CV_Assignment_5_2023.ipynb",
        ["LeNet", "fgsm_attack"],
    )
    assert "LeNet" in symbols and "fgsm_attack" in symbols
    LeNet = symbols["LeNet"]
    fgsm_attack = symbols["fgsm_attack"]

    model = LeNet()
    x = torch.rand(1, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (1, 10)

    # Prediction matches target: perturbation is added and clamped
    pred = logits.argmax(dim=1)
    target = pred.clone()

    epsilon = 0.15
    perturbed = fgsm_attack(model, x.clone(), target, epsilon)

    assert perturbed.shape == (1, 1, 28, 28)
    assert torch.all(perturbed >= 0.0) and torch.all(perturbed <= 1.0)
    # The image should be perturbed
    assert not torch.allclose(perturbed, x)


def test_assignment5_rotations_and_dual_head_net():
    """Verify 4-fold image rotations and dual-head LeNet architecture for SSL pretext task."""
    symbols = extract_symbols_from_notebook(
        "DL4CV_Assignment_5_2023.ipynb",
        [
            "tensor_rot_90",
            "tensor_rot_180",
            "tensor_rot_270",
            "rotate_batch_with_labels",
            "Net",
        ],
    )
    assert "tensor_rot_90" in symbols and "Net" in symbols
    tensor_rot_90 = symbols["tensor_rot_90"]
    tensor_rot_180 = symbols["tensor_rot_180"]
    tensor_rot_270 = symbols["tensor_rot_270"]
    rotate_batch_with_labels = symbols["rotate_batch_with_labels"]
    Net = symbols["Net"]

    # Test rotation group property: 4 x 90-degree rotations equals identity
    img = torch.randn(1, 28, 28)
    rot90 = tensor_rot_90(img)
    rot180 = tensor_rot_90(rot90)
    rot270 = tensor_rot_90(rot180)
    rot360 = tensor_rot_90(rot270)
    assert torch.allclose(img, rot360, atol=1e-5)
    assert torch.allclose(rot180, tensor_rot_180(img), atol=1e-5)
    assert torch.allclose(rot270, tensor_rot_270(img), atol=1e-5)

    # Test rotate_batch_with_labels
    batch = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    rotated_batch = rotate_batch_with_labels(batch, labels)
    assert rotated_batch.shape == (4, 1, 28, 28)

    # Test dual-head network
    net = Net()
    cls_out, ssl_out = net(batch)
    assert cls_out.shape == (4, 10), "Classification head must output 10 digit classes"
    assert ssl_out.shape == (4, 4), "Self-supervised rotation head must output 4 rotation classes"


def test_assignment5_cyclegan_losses():
    """Verify CycleGAN discriminator, generator, and cycle consistency loss functions."""
    symbols = extract_symbols_from_notebook(
        "DL4CV_Assignment_5_2023.ipynb",
        ["discriminator_loss", "generator_loss", "cycle_loss"],
    )
    assert "cycle_loss" in symbols
    cycle_loss = symbols["cycle_loss"]
    discriminator_loss = symbols["discriminator_loss"]
    generator_loss = symbols["generator_loss"]

    orig = torch.randn(8, 784)
    rec = orig.clone()
    assert torch.allclose(cycle_loss(orig, rec), torch.tensor(0.0), atol=1e-6)

    # Non-identical inputs produce positive L1 cycle loss
    rec_distorted = orig + 0.5
    assert cycle_loss(orig, rec_distorted).item() > 0.0

    d_loss = discriminator_loss(torch.randn(8, 1), torch.randn(8, 1))
    g_loss = generator_loss(torch.randn(8, 1))
    assert d_loss.dim() == 0 and d_loss.item() > 0.0
    assert g_loss.dim() == 0 and g_loss.item() > 0.0
