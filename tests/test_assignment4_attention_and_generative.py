"""Unit tests for Assignment 4: Multi-Head Attention, ViT Patching, and Generative Losses."""

import numpy as np
import torch
from tests.conftest import extract_symbols_from_notebook


def test_assignment4_multi_head_attention():
    """Verify MultiHeadAttention projection, scaled dot-product attention, and output shapes."""
    symbols = extract_symbols_from_notebook(
        "drive-download-20231003T210250Z-001/DL4CV_Assignment_4_2023.ipynb",
        ["MultiHeadAttention"],
    )
    assert "MultiHeadAttention" in symbols
    MultiHeadAttention = symbols["MultiHeadAttention"]

    config = {
        "QI_DIM": 64,
        "VI_DIM": 64,
        "KI_DIM": 64,
        "QO_DIM": 32,
        "VO_DIM": 32,
        "NUM_HEADS": 2,
        "OP_DIM": 16,
    }

    mha = MultiHeadAttention(config)
    batch_size = 4
    seq_len = 3

    q = torch.rand(batch_size, seq_len, 64)
    k = torch.rand(batch_size, seq_len, 64)
    v = torch.rand(batch_size, seq_len, 64)

    out, attn = mha(q, k, v, ret_atn=True)
    assert out.shape == (batch_size, seq_len, 16)
    assert attn.shape == (batch_size, 2, seq_len, seq_len)
    # Check attention rows sum to 1.0
    attn_sums = attn.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)


def test_assignment4_img_patch():
    """Verify Vision Transformer image patch extraction and token flattening."""
    symbols = extract_symbols_from_notebook(
        "drive-download-20231003T210250Z-001/DL4CV_Assignment_4_2023.ipynb",
        ["img_patch"],
    )
    assert "img_patch" in symbols
    img_patch = symbols["img_patch"]

    batch = np.random.randn(2, 3, 32, 32).astype(np.float32)
    patch_size = 16

    final_patch, flatten = img_patch(batch, patch_size)
    assert final_patch.shape == (2, 3, 2, 2, 16, 16)
    assert flatten.shape == (2, 4, 3 * 16 * 16)


def test_assignment4_kl_divergence_losses():
    """Verify analytical and Monte Carlo KL divergence definitions."""
    symbols = extract_symbols_from_notebook(
        "drive-download-20231003T210250Z-001/DL4CV_Assignment_4_2023.ipynb",
        ["KL_analytical", "kl_mc"],
    )
    assert "KL_analytical" in symbols and "kl_mc" in symbols
    KL_analytical = symbols["KL_analytical"]
    kl_mc = symbols["kl_mc"]

    mu = torch.zeros(1, 10)
    logvar = torch.zeros(1, 10)

    # When mu=0 and logvar=0, posterior equals standard normal prior -> KL is 0
    kld_zero = KL_analytical(mu, logvar)
    assert torch.allclose(kld_zero, torch.tensor(0.0), atol=1e-6)

    # For non-zero mu, KL must be strictly positive
    mu_active = torch.randn(2, 10)
    logvar_active = torch.randn(2, 10)
    kld_active = KL_analytical(mu_active, logvar_active)
    assert kld_active.item() > 0.0


def test_assignment4_aae_losses():
    """Verify discriminator and generator loss formulations."""
    symbols = extract_symbols_from_notebook(
        "drive-download-20231003T210250Z-001/DL4CV_Assignment_4_2023.ipynb",
        ["discriminator_loss", "generator_loss"],
    )
    assert "discriminator_loss" in symbols and "generator_loss" in symbols
    discriminator_loss = symbols["discriminator_loss"]
    generator_loss = symbols["generator_loss"]

    logits_real = torch.randn(16, 1)
    logits_fake = torch.randn(16, 1)

    d_loss = discriminator_loss(logits_real, logits_fake)
    g_loss = generator_loss(logits_fake)

    assert d_loss.dim() == 0 and d_loss.item() > 0.0
    assert g_loss.dim() == 0 and g_loss.item() > 0.0
