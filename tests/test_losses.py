import torch

from crova.losses import NAMES, PROFILES, gap_loss, losses, objective, reference_order


def test_losses_vanish_at_reference_with_zero_gradient():
    reference = torch.randn(6, 50).to(torch.bfloat16)
    candidate = reference.float().clone().requires_grad_(True)
    values = losses(candidate, reference)
    for name in NAMES:
        assert abs(float(values[name])) < 1e-6, name
        (grad,) = torch.autograd.grad(values[name], candidate, retain_graph=True)
        assert float(grad.abs().max()) < 1e-6, name


def test_losses_positive_away_from_reference():
    reference = torch.randn(4, 40)
    candidate = reference + 0.3 * torch.randn(4, 40)
    assert all(float(v) > 0 for v in losses(candidate, reference).values())


def test_gap_loss_matches_definition():
    torch.manual_seed(1)
    reference, candidate = torch.randn(3, 12), torch.randn(3, 12)
    order = reference_order(reference, 5)
    d = candidate - reference
    expected = []
    for row, q in zip(d, order, strict=True):
        selected = row[q]
        adjacent = (selected[:-1] - selected[1:]).square().sum()
        outside = [j for j in range(12) if j not in q.tolist()]
        boundary = (selected[-1] - row[outside]).square().mean()
        expected.append((adjacent + boundary) / 5)
    torch.testing.assert_close(gap_loss(candidate, reference, order), torch.stack(expected).mean())


def test_reference_order_breaks_ties_by_token_id():
    reference = torch.tensor([[1.0, 3.0, 3.0, 0.0, 3.0, 2.0]])
    assert reference_order(reference, 5).tolist() == [[1, 2, 4, 5, 0]]


def test_profiles_are_normalised():
    for name, weights in PROFILES.items():
        assert len(weights) == 4 and abs(sum(weights) - 1) < 1e-12, name


def test_objective_weights_normalised_losses():
    values = {name: torch.tensor(float(i + 1)) for i, name in enumerate(NAMES)}
    scales = dict.fromkeys(NAMES, 2.0)
    assert float(objective(values, PROFILES["equal"], scales)) == 0.25 * (1 + 2 + 3 + 4) / 2


def test_token_weighting_scales_the_objective_but_not_the_reported_losses():
    from crova.lora import _loss_terms

    head = torch.nn.Linear(8, 20)
    hidden, target = torch.randn(6, 8), torch.randn(6, 20)
    scales = dict.fromkeys(NAMES, 1.0)
    one, terms_one = _loss_terms(head, hidden, target, "cpu", weights=PROFILES["equal"], scales=scales)
    three, terms_three = _loss_terms(head, hidden, target, "cpu", weights=PROFILES["equal"],
                                     scales=scales, multiplier=3.0)
    assert abs(three - 3 * one) < 1e-5 and terms_one == terms_three
