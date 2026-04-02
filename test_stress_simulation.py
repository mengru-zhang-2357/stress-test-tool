import math

import pandas as pd

from stress_simulation import LineItem, _compute_portfolio_metrics, _rebalance_portfolio
from stress_simulation import simulate_portfolio


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


def test_illiquidity_premium_is_applied_to_private_return_path():
    asset_alloc_df = pd.DataFrame(
        [
            {
                "Item": "Buyout",
                "Allocation": 100.0,
                "Beta": 1.0,
                "Monthly Liquidity %": 0.0,
                "Private %": 1.0,
            }
        ]
    )
    liquidity_df = pd.DataFrame([{"Item": "Cash", "Liquidity Order": 1}])
    cash_flows_df = pd.DataFrame(
        [{"Item": "Buyout", "Projection Year": 1, "Capital Call %": 0.0, "Distribution %": 0.0}]
    )

    result = simulate_portfolio(
        asset_alloc_df=asset_alloc_df,
        liquidity_df=liquidity_df,
        cash_flows_df=cash_flows_df,
        baseline_return=0.08,
        baseline_std=0.16,
        illiquidity_premium=0.03,
        annual_dividend=0.0,
        n_years=1,
    )

    # Year-1 market return is fixed at -40%; formula should be:
    # 100 * (1 + 0.6 * -0.4 + 0.03) = 79
    assert math.isclose(float(result.loc[0, "nav_total_pre"]), 79.0, abs_tol=1e-9)


def test_public_returns_clamp_negative_nav_to_zero():
    asset_alloc_df = pd.DataFrame(
        [
            {
                "Item": "Hedge Overlay",
                "Allocation": 100.0,
                "Beta": -5.0,
                "Monthly Liquidity %": 1.0,
                "Private %": 0.0,
            },
            {
                "Item": "Cash",
                "Allocation": 0.0,
                "Beta": 0.0,
                "Monthly Liquidity %": 1.0,
                "Private %": 0.0,
            },
        ]
    )
    liquidity_df = pd.DataFrame([{"Item": "Cash", "Liquidity Order": 1}])
    cash_flows_df = pd.DataFrame(columns=["Item", "Projection Year", "Capital Call %", "Distribution %"])

    # Year 2 market return of +30% would drive the -5 beta sleeve to
    # 100 * (1 - 1.5) = -50 without clamping.
    result = simulate_portfolio(
        asset_alloc_df=asset_alloc_df,
        liquidity_df=liquidity_df,
        cash_flows_df=cash_flows_df,
        annual_dividend=0.0,
        n_years=2,
        scenario_returns={2: 0.30},
    )

    year_2_items = result.loc[result["year"] == 2, "items"].iloc[0]
    assert year_2_items["Hedge Overlay"]["nav"] == 0.0


def test_rebalance_excludes_negative_beta_hedges():
    items = _items(
        LineItem("Cash", 0.0, 0.0, 1.0, 0.0),
        LineItem("Hedge", 20.0, -0.5, 1.0, 0.0),
        LineItem("LowBeta", 40.0, 0.3, 1.0, 0.0),
        LineItem("HighBeta", 40.0, 1.0, 1.0, 0.0),
    )

    _rebalance_portfolio(items, beta_start=0.90, tolerance=0.01)

    # Negative-beta hedge sleeve should be untouched by rebalancing.
    assert math.isclose(items["Hedge"].nav, 20.0, abs_tol=1e-9)
    _, beta_after, _ = _compute_portfolio_metrics(items)
    assert math.isclose(beta_after, 0.90, abs_tol=0.01)
