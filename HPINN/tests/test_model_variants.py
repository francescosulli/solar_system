import torch

from solsys_emulator.model import EmulatorModel


def test_residual_position_only_model_with_body_embeddings():
    model = EmulatorModel(
        num_bodies=3,
        state_mode="position_only",
        backbone_type="residual",
        hidden_dim=64,
        num_layers=3,
        fourier_features=8,
        min_frequency=0.1,
        max_frequency=16.0,
        frequency_spacing="log",
        head_layers=2,
        head_hidden_dim=32,
        body_embedding_dim=12,
        interaction_layers=2,
        interaction_hidden_dim=48,
        use_layer_norm=True,
    )
    t = torch.linspace(-1.0, 1.0, steps=5, requires_grad=True)
    out = model(t)
    assert out.shape == (5, 3, 3)
    loss = out.square().mean()
    loss.backward()
    assert model.body_embeddings is not None
    assert model.body_embeddings.weight.grad is not None
    assert torch.isfinite(out).all()


def test_hybrid_correction_model_exposes_components():
    model = EmulatorModel(
        num_bodies=4,
        state_mode="position_only",
        backbone_type="residual",
        hidden_dim=48,
        num_layers=2,
        fourier_features=6,
        min_frequency=0.1,
        max_frequency=12.0,
        frequency_spacing="log",
        head_layers=2,
        head_hidden_dim=24,
        body_embedding_dim=8,
        interaction_layers=1,
        interaction_hidden_dim=24,
        hybrid_correction=True,
        correction_layers=2,
        correction_hidden_dim=16,
        correction_init_scale=0.02,
        use_layer_norm=True,
    )
    t = torch.linspace(-1.0, 1.0, steps=7, requires_grad=True)
    final_out, base_out, correction_out = model.forward_components(t)
    assert final_out.shape == (7, 4, 3)
    assert base_out.shape == (7, 4, 3)
    assert correction_out.shape == (7, 4, 3)
    assert torch.allclose(final_out, base_out + correction_out, atol=1e-6)
    loss = final_out.square().mean() + correction_out.square().mean()
    loss.backward()
    assert model.correction_gain_unconstrained is not None
    assert model.correction_gain_unconstrained.grad is not None
