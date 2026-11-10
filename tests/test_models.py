import json
from pathlib import Path

import numpy as np
import pytest
import torch

from selective_rul.backbones import MambaReferenceBlock, pinball_loss, selective_scan_reference
from selective_rul.experiment import check_config
from selective_rul.model import HEAD_NAMES, MODEL_NAMES, RULModel, build_model, predict_details

ROOT = Path(__file__).resolve().parents[1]


def config(model="tcn", head="point", window=30):
    cfg = json.loads((ROOT / "configs/smoke.json").read_text())
    return dict(cfg, model=model, head=head, window=window, hidden=8, layers=1)


@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize("head", HEAD_NAMES)
def test_all_models_train_and_reload_without_losing_predictions(model, head, tmp_path):
    torch.manual_seed(1)
    cfg = config(model, head)
    network = build_model(31, cfg)
    x = torch.randn(3, 30, 31)
    output = network.forward_outputs(x)
    assert output["point"].shape == (3,)
    assert ((output["point"] > 0) & (output["point"] < 125)).all()
    loss = ((output["point"] - 80) / 125).square().mean()
    if head == "quantile":
        q = output["quantiles"]
        assert q.shape == (3, 3)
        assert (q[:, 1:] >= q[:, :-1]).all() and (q >= 0).all()
        loss = loss + pinball_loss(q, torch.tensor([180.0, 60.0, 5.0])).mean() / 125
    loss.backward()
    gradients = [p.grad for p in network.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    torch.save(network.state_dict(), path)
    loaded = build_model(31, cfg)
    loaded.load_state_dict(torch.load(path, weights_only=True))
    before = predict_details([network], x.numpy(), "cpu")
    after = predict_details([loaded], x.numpy(), "cpu")
    np.testing.assert_array_equal(before[0], after[0])
    if head == "quantile":
        np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_backbones_support_longer_observed_history(model):
    cfg = config(model, window=60)
    check_config(cfg)
    network = build_model(31, cfg).eval()
    with torch.no_grad():
        assert network(torch.randn(2, 60, 31)).shape == (2,)


def test_patch_tokens_include_current_cycle_and_are_not_single_token():
    network = build_model(31, config("patch_transformer")).eval()
    assert network.n_patches == 5
    first = torch.randn(1, 30, 31)
    changed = first.clone()
    changed[:, -1] += torch.arange(31)
    assert not torch.allclose(network(first), network(changed))
    with pytest.raises(ValueError, match="tiling"):
        check_config(config("patch_transformer", window=32))


def test_mamba_scan_matches_closed_form_and_block_cannot_read_future():
    u = torch.ones(1, 4, 1)
    output = selective_scan_reference(u, u, -torch.ones(1, 1), u, u, torch.zeros(1))
    expected = torch.tensor([sum(np.exp(-j) for j in range(t + 1)) for t in range(4)])
    torch.testing.assert_close(output.flatten(), expected.float())
    block = MambaReferenceBlock(8, dropout=0).eval()
    first = torch.randn(2, 30, 8)
    second = first.clone()
    second[:, 15:] += 100
    torch.testing.assert_close(block(first)[:, :15], block(second)[:, :15])


def test_raw_quantiles_can_exceed_regression_cap_and_use_raw_targets():
    network = build_model(31, config(head="quantile"))
    with torch.no_grad():
        network.quantile_head.weight.zero_()
        network.quantile_head.bias.fill_(3)
    output = network.forward_outputs(torch.ones(2, 30, 31))
    assert (output["point"] <= 125).all()
    assert (output["quantiles"] > 125).all()
    q = torch.tensor([[10.0, 20.0, 30.0]])
    assert pinball_loss(q, torch.tensor([40.0])).item() == pytest.approx(22 / 3)
    assert (
        pinball_loss(q, torch.tensor([170.0])).item()
        > pinball_loss(q, torch.tensor([125.0])).item()
    )


def test_legacy_tcn_state_dict_is_compatible():
    legacy = RULModel(31, hidden=16, dropout=0.1)
    cfg = config()
    cfg["hidden"] = 16
    restored = build_model(31, cfg)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    legacy.eval()
    restored.eval()
    x = torch.randn(2, 30, 31)
    torch.testing.assert_close(legacy(x), restored(x))


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_softplus_readout_is_unbounded_and_default_stays_capped(model):
    torch.manual_seed(1)
    capped = build_model(31, config(model))
    uncapped = build_model(31, dict(config(model), point_readout="softplus", point_target="raw"))
    assert capped.readout == "sigmoid" and uncapped.readout == "softplus"
    uncapped.load_state_dict(capped.state_dict())  # identical parameter keys
    features = torch.zeros(1, uncapped.head.in_features)
    with torch.no_grad():
        uncapped.head.bias.fill_(10.0)
        capped.head.bias.fill_(10.0)
    assert float(uncapped.regress(features)) > 125
    assert float(capped.regress(features)) < 125


def test_raw_target_requires_softplus_readout():
    with pytest.raises(ValueError, match="softplus"):
        check_config(dict(config(), point_target="raw"))
    check_config(dict(config(), point_target="raw", point_readout="softplus"))
