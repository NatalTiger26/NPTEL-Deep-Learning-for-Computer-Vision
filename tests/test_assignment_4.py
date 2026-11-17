"""Unit tests for Assignment 4: Self-Attention, Vision Transformer Patching, and VAE/AAE."""

import math
from typing import Dict, Tuple, Union
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Attention layer."""

    def __init__(self, config: Dict[str, int]) -> None:
        super(MultiHeadAttention, self).__init__()
        self.op_dim: int = config['OP_DIM']
        self.n_heads: int = config['NUM_HEADS']
        self.query_dim: int = config['QO_DIM']
        self.value_dim: int = config['VO_DIM']

        self.head_dim: int = self.query_dim // self.n_heads
        self.v_head_dim: int = self.value_dim // self.n_heads

        self.QTrans: nn.Linear = nn.Linear(config['QI_DIM'], self.query_dim)
        self.KTrans: nn.Linear = nn.Linear(config['KI_DIM'], self.query_dim)
        self.VTrans: nn.Linear = nn.Linear(config['VI_DIM'], self.value_dim)
        self.OTrans: nn.Linear = nn.Linear(self.value_dim, self.op_dim)

        self.scale: float = math.sqrt(float(self.head_dim))

    def forward(
        self,
        q_inp: torch.Tensor,
        k_inp: torch.Tensor,
        v_inp: torch.Tensor,
        ret_atn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert q_inp.shape[0] == k_inp.shape[0] == v_inp.shape[0]
        batch_size: int = q_inp.shape[0]
        seq_length: int = q_inp.shape[1]

        Q = self.QTrans(q_inp).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.KTrans(k_inp).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.VTrans(v_inp).view(batch_size, -1, self.n_heads, self.v_head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)

        attended = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = self.OTrans(attended.view(batch_size, seq_length, -1))

        if ret_atn:
            return out, attention
        return out


class LinearAttentionBlock(nn.Module):
    """Linear attention block combining spatial local features and global context."""

    def __init__(self, in_features: int, normalize_attn: bool = True) -> None:
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.W_g = nn.Conv2d(in_features, in_features, kernel_size=1, padding=0)
        self.W_x = nn.Conv2d(in_features, in_features, kernel_size=1, padding=0)
        self.psi = nn.Conv2d(in_features, 1, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, l: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g1 = self.W_g(g)
        x1 = self.W_x(l)
        f = self.relu(g1 + x1)
        psi = self.psi(f)
        if self.normalize_attn:
            b, c, h, w = psi.shape
            psi_flat = psi.view(b, -1)
            psi_norm = torch.softmax(psi_flat, dim=-1).view(b, c, h, w)
        else:
            psi_norm = torch.sigmoid(psi)
        attended = l * psi_norm
        return attended, psi_norm


def img_patch(batch: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Partition an image batch into spatial patches (Vision Transformer tokenizer)."""
    batch_size, c, h, w = batch.shape
    n_h = h // patch_size
    n_w = w // patch_size

    patches = batch.reshape(batch_size, c, n_h, patch_size, n_w, patch_size)
    final_patch = patches.transpose(0, 1, 2, 4, 3, 5)
    flatten = final_patch.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size, n_h * n_w, c * patch_size * patch_size)
    return final_patch, flatten


def KL_analytical(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Exact closed-form KL divergence with standard Normal prior."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def kl_mc(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Monte Carlo empirical estimate of KL divergence."""
    q = torch.distributions.Normal(mu, std)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    return torch.sum(q.log_prob(z) - p.log_prob(z), dim=-1)


def discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy discriminator loss with real=1, fake=0 targets."""
    real_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)
    return F.binary_cross_entropy_with_logits(logits_real, real_labels) + F.binary_cross_entropy_with_logits(
        logits_fake, fake_labels
    )


def generator_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy generator loss tricking discriminator."""
    real_labels = torch.ones_like(logits_fake)
    return F.binary_cross_entropy_with_logits(logits_fake, real_labels)


def test_multihead_attention_shapes_and_device() -> None:
    """Test MultiHeadAttention forward pass, output shape, and attention matrix shape."""
    config = {
        'QI_DIM': 64,
        'VI_DIM': 64,
        'KI_DIM': 64,
        'QO_DIM': 32,
        'VO_DIM': 32,
        'OP_DIM': 32,
        'NUM_HEADS': 8,
    }
    mha = MultiHeadAttention(config)

    batch_size = 4
    seq_len = 2
    q = torch.rand(batch_size, seq_len, 64)
    k = torch.rand(batch_size, seq_len, 64)
    v = torch.rand(batch_size, seq_len, 64)

    out, attn = mha(q, k, v, ret_atn=True)
    assert out.shape == (batch_size, seq_len, 32)
    assert attn.shape == (batch_size, 8, seq_len, seq_len)
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)


def test_linear_attention_block() -> None:
    """Test spatial LinearAttentionBlock."""
    att = LinearAttentionBlock(in_features=16, normalize_attn=True)
    l = torch.rand(4, 16, 64, 64)
    g = torch.rand(4, 16, 64, 64)

    attended, psi_norm = att(l, g)
    assert attended.shape == (4, 16, 64, 64)
    assert psi_norm.shape == (4, 1, 64, 64)


def test_img_patch_tokenization() -> None:
    """Test Vision Transformer image patching and tokenization."""
    img = np.random.randn(1, 3, 512, 512)
    patch_size = 16
    final_patch, flatten = img_patch(img, patch_size)

    n_h = 512 // 16  # 32
    n_w = 512 // 16  # 32
    num_patches = n_h * n_w  # 1024
    patch_dim = 3 * 16 * 16  # 768

    assert final_patch.shape == (1, 3, 32, 32, 16, 16)
    assert flatten.shape == (1, num_patches, patch_dim)


def test_kl_analytical_and_monte_carlo() -> None:
    """Test agreement between analytical and Monte Carlo sampled KL divergence."""
    torch.manual_seed(1)
    mu = torch.rand((1, 40))
    logvar = torch.rand((1, 40))
    std = torch.exp(logvar / 2.0)

    kl_ana = KL_analytical(mu, logvar).item()

    q_dist = torch.distributions.Normal(mu, std)
    z = q_dist.rsample(sample_shape=(10000,))
    kl_monte = kl_mc(z, mu, std).mean().item()

    # Monte Carlo estimate converges to exact analytical KL
    assert np.isclose(kl_ana, kl_monte, rtol=0.05)


def test_aae_loss_functions() -> None:
    """Test Adversarial Autoencoder loss calculations."""
    logits_real = torch.tensor([2.5, 1.8, 3.0])
    logits_fake = torch.tensor([-1.2, -0.8, -2.1])

    d_loss = discriminator_loss(logits_real, logits_fake)
    g_loss = generator_loss(logits_fake)

    assert d_loss.item() > 0.0
    assert g_loss.item() > 0.0
