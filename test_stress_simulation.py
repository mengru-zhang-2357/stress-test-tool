import math

from stress_simulation import LineItem, _compute_portfolio_metrics, _rebalance_portfolio


def _items(*line_items):
    return {li.name: li for li in line_items}


def test_rebalance_beta_increase_avoids_overshoot_with_high_beta_destination():
    items = _items(
        LineItem("Cash", 20.0, 0.0, 1.0, 0.0),
        LineItem("Bonds", 40.0, 0.2, 1.0, 0.0),
        LineItem("LeveredEquity", 40.0, 1.5, 1.0, 0.0),
    )

    _rebalance_portfolio(items, beta_start=0.90, tolerance=0.02)

    _, beta_after, _ = _compute_portfolio_metrics(items)
    assert beta_after <= 0.92
    assert math.isclose(beta_after, 0.90, abs_tol=0.02)


def test_rebalance_beta_increase_handles_narrow_beta_spread_without_undershoot():
    items = _items(
        LineItem("Cash", 0.0, 0.0, 1.0, 0.0),
        LineItem("LowBeta", 50.0, 0.58, 1.0, 0.0),
        LineItem("SlightlyHigherBeta", 50.0, 0.63, 1.0, 0.0),
    )

    _rebalance_portfolio(items, beta_start=0.62, tolerance=0.005)

    _, beta_after, _ = _compute_portfolio_metrics(items)
    assert math.isclose(beta_after, 0.62, abs_tol=0.005)


def test_rebalance_converges_within_tolerance_when_liquidity_is_sufficient():
    items = _items(
        LineItem("Cash", 0.0, 0.0, 1.0, 0.0),
        LineItem("HighBeta1", 45.0, 1.3, 1.0, 0.0),
        LineItem("HighBeta2", 45.0, 1.1, 1.0, 0.0),
        LineItem("LowBeta", 10.0, 0.1, 1.0, 0.0),
    )

    _rebalance_portfolio(items, beta_start=0.70, tolerance=0.01)

    _, beta_after, _ = _compute_portfolio_metrics(items)
    assert math.isclose(beta_after, 0.70, abs_tol=0.01)


def test_rebalance_stops_gracefully_when_liquidity_is_insufficient():
    items = _items(
        LineItem("Cash", 0.0, 0.0, 1.0, 0.0),
        LineItem("IlliquidLow", 80.0, 0.1, 0.0, 0.0),
        LineItem("SmallLiquidLow", 15.0, 0.2, 0.2, 0.0),
        LineItem("HighBeta", 5.0, 1.4, 1.0, 0.0),
    )

    _, beta_before, _ = _compute_portfolio_metrics(items)
    _rebalance_portfolio(items, beta_start=0.80, tolerance=0.01)
    _, beta_after, _ = _compute_portfolio_metrics(items)

    # Beta should improve, but remain outside tolerance because available
    # liquid low-beta capital is limited.
    assert beta_after > beta_before
    assert beta_after < 0.79
    assert items["SmallLiquidLow"].liquid_amount() == 0.0
