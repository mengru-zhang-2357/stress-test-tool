"""
stress_simulation.py
--------------------

This module contains functions to perform portfolio stress testing and
simulation logic.  The core function ``simulate_portfolio`` implements
the mechanics described by the user, including:

* Applying an initial equity market shock (–40% in year 1) that flows
  through each line item according to its beta exposure.
* Incorporating private asset cash flows (capital calls and
  distributions) using the specified formula and moving the net cash
  flows to or from a cash/fixed‑income bucket.
* Handling annual dividends by sourcing cash from line items according
  to a user‑defined liquidity waterfall.  Each pass down the waterfall
  may tap up to 50 % of an asset’s monthly liquid balance.
* Generating subsequent years’ returns either from a user‑provided
  scenario or by sampling from a normal distribution with mean and
  standard deviation supplied in the inputs.
* Rebalancing the portfolio whenever the weighted average beta deviates
  by more than 0.02 from its initial value.  Rebalancing uses only
  monthly liquid assets: if the beta is too low, the function buys
  additional exposure by shifting liquid capital from low‑beta assets to
  the highest‑beta asset; if beta is too high, it sells high‑beta
  assets to cash/fixed income.  Monthly liquidity and private
  percentages are updated after every transaction.

The simulation functions are independent of any user interface.  A
Shiny application can import ``simulate_portfolio`` and other helper
functions from this module to perform computations based on user
inputs.

Note:  Percentages in the input data (e.g. monthly liquidity, private
percent, capital call percent, distribution percent) may be supplied
either as decimals (0–1) or percentages (0–100).  The helper
``_to_fraction`` normalizes these values to decimals when needed.

"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable


def _to_fraction(value: float) -> float:
    """Normalize a percentage‐like value to a fraction.

    Values greater than 1 are interpreted as percentages (e.g. 50 -> 0.5).
    None or NaN inputs return 0.0.

    Args:
        value: The raw value.

    Returns:
        A float between 0 and 1.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if abs(value) > 1.0:
        return value / 100.0
    return value


@dataclass
class LineItem:
    """Represent a single line item in the portfolio.

    Attributes:
        name:  Name of the line item.
        nav:   Net asset value (dollar amount) for this item.
        beta:  Beta assumption for market sensitivity.
        monthly_liq: Monthly liquid portion of the NAV (fraction 0–1).
        private: Private portion of the NAV (fraction 0–1).

    The class exposes convenience properties for computing the monetary
    amounts of the monthly liquid and private portions and for updating
    these portions after trades or sales.
    """

    name: str
    nav: float
    beta: float
    monthly_liq: float  # fraction (0–1)
    private: float      # fraction (0–1)

    def liquid_amount(self) -> float:
        """Return the dollar amount of the monthly liquid portion."""
        return self.nav * self.monthly_liq

    def private_amount(self) -> float:
        """Return the dollar amount of the private portion."""
        return self.nav * self.private

    def update_after_sale(self, sale_amount: float) -> None:
        """Reduce NAV and liquid portion after selling from liquid assets.

        Args:
            sale_amount: The amount of NAV sold (taken from the liquid
                portion).  It must not exceed the current liquid amount.
        """
        if sale_amount < 0:
            raise ValueError("sale_amount cannot be negative")
        liquid_amt = self.liquid_amount()
        if sale_amount > liquid_amt + 1e-12:
            raise ValueError(
                f"Attempting to sell {sale_amount:.2f} from {self.name}, "
                f"but only {liquid_amt:.2f} is available for immediate sale."
            )
        # Reduce NAV and liquid NAV accordingly.  Private amount remains
        # unchanged because we only sell from the liquid portion.
        if self.nav > 0:
            liquid_fraction_sold = sale_amount / self.nav
        else:
            liquid_fraction_sold = 0
        # The sale amount reduces NAV in total.  Because only the liquid
        # portion is reduced, the private NAV remains constant.
        private_nav_before = self.private_amount()
        self.nav -= sale_amount
        if self.nav <= 0:
            # Entire position sold.  Reset to zero.
            self.nav = 0.0
            self.monthly_liq = 0.0
            self.private = 0.0
        else:
            # Update monthly_liq and private fractions based on the new
            # NAV and the unchanged private NAV.
            # new_private = private_nav_before
            self.private = private_nav_before / self.nav
            # liquid NAV after sale = nav * monthly_liq
            # monthly_liq fraction = liquid NAV after sale / nav
            liquid_nav_after = max(liquid_amt - sale_amount, 0.0)
            self.monthly_liq = liquid_nav_after / self.nav
            # ensure fractions are within [0,1]
            self.private = max(min(self.private, 1.0), 0.0)
            self.monthly_liq = max(min(self.monthly_liq, 1.0), 0.0)

    def add_investment(self, amount: float, liquid: bool = True) -> None:
        """Increase NAV and adjust liquidity when buying this asset.

        Args:
            amount: Dollar amount to add.
            liquid: If True, the purchased portion is considered 100 % monthly
                liquid; otherwise it is considered 0 % monthly liquid (but
                counted as private).  The private fraction grows
                accordingly.
        """
        if amount < 0:
            raise ValueError("add_investment requires a non‑negative amount")
        if amount == 0:
            return
        # Determine the private amount before purchase.
        private_nav_before = self.private_amount()
        liquid_nav_before = self.liquid_amount()
        # Update NAV
        self.nav += amount
        # Determine how the new amount affects liquidity and private splits
        if liquid:
            # New money is 100% liquid.
            liquid_nav_new = liquid_nav_before + amount
            private_nav_new = private_nav_before
        else:
            # New money goes into the private bucket.
            liquid_nav_new = liquid_nav_before
            private_nav_new = private_nav_before + amount
        # Update fractions
        if self.nav > 0:
            self.monthly_liq = liquid_nav_new / self.nav
            self.private = private_nav_new / self.nav
        else:
            self.monthly_liq = 0.0
            self.private = 0.0
        # clamp within [0,1]
        self.monthly_liq = max(min(self.monthly_liq, 1.0), 0.0)
        self.private = max(min(self.private, 1.0), 0.0)


def _initialize_line_items(asset_alloc_df: pd.DataFrame) -> Dict[str, LineItem]:
    """Create a mapping of line items from the input asset allocation DataFrame.

    The input DataFrame must have columns: ``Item``, ``Allocation``, ``Beta``,
    ``Monthly Liquidity %`` and ``Private %``.  Percent columns can be
    specified as decimals (0–1) or percentages (0–100).  The ``Allocation``
    column is interpreted as the starting net asset value in dollars.  Beta
    values are interpreted as floats.

    Args:
        asset_alloc_df: DataFrame containing the asset allocation.

    Returns:
        Dictionary mapping item names to ``LineItem`` objects.
    """
    items: Dict[str, LineItem] = {}
    for _, row in asset_alloc_df.iterrows():
        name = str(row['Item']).strip()
        if not name:
            continue
        nav = float(row['Allocation']) if not pd.isna(row['Allocation']) else 0.0
        beta = float(row['Beta']) if not pd.isna(row['Beta']) else 0.0
        monthly_liq = _to_fraction(float(row['Monthly Liquidity %']))
        private = _to_fraction(float(row['Private %']))
        # Ensure fractions do not exceed 1 when combined.  If they do,
        # normalize them proportionally to 1.
        total_frac = monthly_liq + private
        if total_frac > 1.0:
            monthly_liq = monthly_liq / total_frac
            private = private / total_frac
        items[name] = LineItem(name=name, nav=nav, beta=beta,
                               monthly_liq=monthly_liq, private=private)
    # Guarantee a cash/fixed income bucket exists.  Identify existing cash or
    # fixed income items by beta close to zero.  If none exist, create one
    # with zero NAV.
    cash_candidates = [li for li in items.values() if abs(li.beta) < 1e-6]
    if not cash_candidates:
        # Create an explicit cash bucket.  Beta=0, 100% liquid, 0% private.
        items['Cash'] = LineItem(name='Cash', nav=0.0, beta=0.0,
                                 monthly_liq=1.0, private=0.0)
    return items


def _initialize_liquidity_order(
    liquidity_df: pd.DataFrame, items: Dict[str, LineItem]
) -> List[str]:
    """Determine the liquidity waterfall order from the input liquidity DataFrame.

    The liquidity DataFrame must have columns ``Item`` and ``Liquidity Order``.
    Items may appear in the DataFrame even if they are not present in the
    asset allocation DataFrame; such items are ignored.  Any item in the
    asset allocation but missing from the liquidity DataFrame will be
    appended to the end of the order with a high order number.  Cash is
    always placed at the start of the order if not already specified.

    Args:
        liquidity_df: DataFrame containing the liquidity waterfall.
        items: Mapping of item names to ``LineItem`` objects.

    Returns:
        List of item names ordered from most liquid (highest priority) to
        least.
    """
    order = []
    # Filter only valid items present in our portfolio
    for _, row in liquidity_df.iterrows():
        name = str(row['Item']).strip()
        if name in items:
            order_rank = row.get('Liquidity Order', None)
            try:
                order_rank = float(order_rank)
            except Exception:
                order_rank = None
            order.append((name, order_rank))
    # Add remaining items not specified.  They will have None order and
    # will be sorted at the end by name.
    for name in items.keys():
        if name not in [n for n, _ in order]:
            order.append((name, None))
    # Ensure cash bucket is first in order.  If 'Cash' exists and not
    # already first, insert it at the beginning.
    order_names = [n for n, _ in order]
    if 'Cash' in items and 'Cash' not in order_names:
        order.insert(0, ('Cash', -math.inf))
    # Sort primarily by Liquidity Order (ascending) then by name
    order_sorted = sorted(order, key=lambda x: (math.inf if x[1] is None else x[1], x[0]))
    return [name for name, _ in order_sorted]


def _initialize_cash_flows(cash_flows_df: pd.DataFrame) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Convert the cash flow projection table into a nested dictionary.

    The returned mapping is ``{item: {year: (call_pct, dist_pct), ...}, ...}``,
    where ``call_pct`` and ``dist_pct`` are fractions (0–1).  Years not
    present in the input default to 0.

    Args:
        cash_flows_df: DataFrame with columns ``Item``, ``Projection Year``,
            ``Capital Call %``, and ``Distribution %``.

    Returns:
        Nested dictionary of capital call and distribution percentages.
    """
    flows: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for _, row in cash_flows_df.iterrows():
        name = str(row['Item']).strip()
        if not name:
            continue
        year = int(row['Projection Year']) if not pd.isna(row['Projection Year']) else 0
        call_pct = _to_fraction(float(row['Capital Call %']))
        dist_pct = _to_fraction(float(row['Distribution %']))
        if name not in flows:
            flows[name] = {}
        flows[name][year] = (call_pct, dist_pct)
    return flows


def _compute_portfolio_metrics(items: Dict[str, LineItem]) -> Tuple[float, float, float]:
    """Compute total NAV, weighted average beta and weighted average private %.

    Args:
        items: Mapping of line items.

    Returns:
        Tuple of (total_nav, beta_total, private_total).  If total NAV is
        zero, beta_total and private_total are set to zero.
    """
    total_nav = sum(li.nav for li in items.values())
    if total_nav <= 0:
        return 0.0, 0.0, 0.0
    beta_total = sum(li.nav * li.beta for li in items.values()) / total_nav
    private_total = sum(li.nav * li.private for li in items.values()) / total_nav
    return total_nav, beta_total, private_total


def _apply_private_cash_flows(
    items: Dict[str, LineItem],
    cash_flows: Dict[str, Dict[int, Tuple[float, float]]],
    year: int,
    baseline_return: float,
    market_return: float,
    cash_item_name: str,
) -> None:
    """Apply capital calls and distributions for private assets in a given year.

    For each item present in ``cash_flows``, compute the call and
    distribution amounts as a fraction of the starting NAV.  Adjust the
    item’s NAV using the formula in the specification, and send the
    net cash flow to or from the cash bucket.  Items not found in
    ``cash_flows`` are treated as public and are not affected by calls
    or distributions.

    Args:
        items: Mapping of line items.  Modified in place.
        cash_flows: Nested dictionary of cash flow fractions by item and year.
        year: Current projection year.
        baseline_return: Baseline return assumption (e.g. 0.08).
        market_return: The equity market return for this year (e.g. –0.40 for
            year 1 shock or a normal draw for subsequent years).
        cash_item_name: Name of the cash/fixed income bucket.
    """
    # Determine which item acts as the cash bucket
    cash_li = items[cash_item_name]
    for name, li in items.items():
        # Skip cash bucket itself
        if name == cash_item_name:
            continue
        # Only items that show up in the cash flow table follow the
        # private cash-flow branch.
        if name not in cash_flows:
            continue
        # Starting NAV for call/distribution calculations
        starting_nav = li.nav
        # Skip items with no NAV
        if starting_nav <= 0:
            continue
        # Retrieve call and distribution percentages for this item/year
        if name in cash_flows and year in cash_flows[name]:
            call_pct, dist_pct = cash_flows[name][year]
        else:
            call_pct, dist_pct = 0.0, 0.0
        # Compute call and distribution amounts based on the starting NAV
        call_amt = starting_nav * call_pct
        dist_amt = starting_nav * dist_pct
        # Apply return shock to NAV (beta * market_return) for this item
        shock_return = li.beta * market_return
        # Diff is item-specific: baseline return minus this item's applied shock return.
        diff = baseline_return - shock_return
        nav_after_return = starting_nav * (1.0 + 0.6 * shock_return)
        # Adjust for calls and distributions
        nav_after_cf = nav_after_return + call_amt * (1.0 + diff) - dist_amt * (1.0 + 2.0 * diff)
        # Compute net cash flow
        net_cf = call_amt * (1.0 + diff) - dist_amt * (1.0 + 2.0 * diff)
        # Update the line item’s NAV directly
        li.nav = nav_after_cf
        # Move cash to/from the cash bucket
        # Positive net_cf indicates a call (cash out of the cash bucket)
        # Negative net_cf indicates distributions (cash added to the cash bucket)
        cash_li.nav -= net_cf
    # After adjusting cash flows, ensure the cash bucket has nonnegative NAV.
    # If negative, set to zero; any negative cash balance can be interpreted
    # as short‑term borrowing, but for this exercise we clamp at zero.
    if cash_li.nav < 0:
        cash_li.nav = 0.0


def _apply_public_returns(
    items: Dict[str, LineItem],
    market_return: float,
    cash_item_name: str,
    private_cash_flow_items: Optional[Iterable[str]] = None,
) -> None:
    """Apply market return to public (non‑private) assets.

    This function multiplies the NAV of each non‑cash, non‑private item by
    ``(1 + beta * market_return)`` to reflect the equity shock.  Private
    assets are excluded here because their return shock and cash‑flow
    adjustments are handled in ``_apply_private_cash_flows``.  The cash
    bucket is left unchanged.

    Args:
        items: Mapping of line items.
        market_return: Equity market return for the year.
        cash_item_name: Name of the cash bucket.
    """
    cf_item_set = set(private_cash_flow_items or [])
    for name, li in items.items():
        # Skip the cash bucket and assets handled by the private cash-flow branch.
        if name == cash_item_name or name in cf_item_set:
            continue
        # Apply return shock: Beta times market return
        shock_return = li.beta * market_return
        li.nav *= (1.0 + shock_return)


def _apply_dividend(
    items: Dict[str, LineItem], liquidity_order: List[str], dividend_amount: float
) -> None:
    """Deduct a dividend from the portfolio using the liquidity waterfall.

    The dividend is sourced from assets according to the provided order.  In
    each pass down the waterfall, up to 50 % of each asset’s monthly
    liquid amount may be sold.  If the dividend cannot be met in a
    single pass, the function loops back to the top and takes another
    50 % of the remaining liquid amount, continuing until the dividend
    is satisfied or all monthly liquid capital is exhausted.

    Args:
        items: Mapping of line items to modify in place.
        liquidity_order: List of item names sorted from highest to lowest
            priority.  The cash bucket should appear first if desired.
        dividend_amount: Amount of cash to distribute.  It should be
            non‑negative.
    """
    if dividend_amount <= 0:
        return
    remaining = dividend_amount
    # Prepare a dictionary of remaining liquid amounts for each item
    liquid_left: Dict[str, float] = {name: items[name].liquid_amount() for name in liquidity_order}
    # Continue taking from the waterfall until the dividend is satisfied
    while remaining > 1e-12:
        any_sold = False
        for name in liquidity_order:
            li = items[name]
            # Recompute available liquid amount (half of remaining liquid)
            available = 0.5 * liquid_left.get(name, 0.0)
            if available <= 0:
                continue
            amt = min(available, remaining)
            if amt > 0:
                # Sell from this item
                li.update_after_sale(amt)
                liquid_left[name] -= amt
                remaining -= amt
                any_sold = True
            if remaining <= 1e-12:
                break
        if not any_sold:
            # No additional liquidity available
            break
    # If remaining > 0, dividend could not be fully satisfied.  For
    # consistency we simply ignore the unfunded portion (the portfolio
    # retains it).  Alternatively, one could reduce cash further or raise
    # liquidity by selling non‑monthly liquid positions, but this is
    # beyond the current spec.
    return


def _rebalance_portfolio(
    items: Dict[str, LineItem], beta_start: float, tolerance: float = 0.02
) -> None:
    """Rebalance the portfolio if the beta deviates beyond a tolerance.

    The function compares the current weighted average beta to the
    starting beta.  If the difference in absolute value exceeds
    ``tolerance``, it shifts liquid capital between high‑beta and
    low‑beta assets to bring the portfolio beta back within the band.
    Only monthly liquid assets are used for rebalancing.  Private
    allocations are unaffected.

    Args:
        items: Mapping of line items.
        beta_start: The starting portfolio beta.
        tolerance: The maximum allowed deviation from ``beta_start``.  If
            the current beta lies outside ``beta_start ± tolerance``, the
            function rebalances to exactly ``beta_start``.
    """
    # Compute current beta and total NAV
    total_nav, beta_current, _ = _compute_portfolio_metrics(items)
    if total_nav <= 0:
        return
    diff = beta_start - beta_current
    if abs(diff) <= tolerance:
        return
    # Determine the amount of NAV to move
    nav_change = abs(diff) * total_nav
    if diff > 0:
        # Current beta is below target: increase beta by buying high‑beta
        # assets using liquid capital from low‑beta assets.
        # Identify the highest beta asset (excluding cash) to buy.
        non_cash_items = [li for li in items.values() if li.name != 'Cash' and li.nav > 0]
        if not non_cash_items:
            return
        target_li = max(non_cash_items, key=lambda li: li.beta)
        # Source capital from liquid low‑beta assets (including cash)
        # sorted by increasing beta
        source_items = sorted(non_cash_items + [items['Cash']], key=lambda li: li.beta)
        # We will allocate nav_change amount to the target item
        remaining = nav_change
        # Collect from sources until we have enough
        for src in source_items:
            if src is target_li:
                continue
            # available liquid amount in this source
            available = src.liquid_amount()
            amt = min(available, remaining)
            if amt > 0:
                src.update_after_sale(amt)
                remaining -= amt
            if remaining <= 1e-12:
                break
        if remaining > 1e-6:
            # Not enough liquidity; limit the purchase to what was sold
            nav_change -= remaining
        # Add the purchased amount to the target item as liquid
        if nav_change > 0:
            target_li.add_investment(nav_change, liquid=True)
    else:
        # Current beta is above target: decrease beta by selling high‑beta
        # assets and sending proceeds to cash.
        high_beta_items = [li for li in items.values() if li.name != 'Cash' and li.nav > 0]
        if not high_beta_items:
            return
        target_li = max(high_beta_items, key=lambda li: li.beta)
        remaining = nav_change
        # Sell from high‑beta items (starting with the highest beta) until
        # we accumulate nav_change; proceeds go to cash.
        cash_li = items['Cash']
        for src in sorted(high_beta_items, key=lambda li: li.beta, reverse=True):
            # available liquid amount to sell
            available = src.liquid_amount()
            amt = min(available, remaining)
            if amt > 0:
                src.update_after_sale(amt)
                cash_li.add_investment(amt, liquid=True)
                remaining -= amt
            if remaining <= 1e-12:
                break
        # If not enough was sold, rebalancing is limited by available liquidity.
    return


def simulate_portfolio(
    asset_alloc_df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    cash_flows_df: pd.DataFrame,
    baseline_return: float = 0.08,
    baseline_std: float = 0.16,
    annual_dividend: float = 0.0,
    dividend_is_percent: bool = False,
    n_years: int = 10,
    scenario_returns: Optional[Dict[int, float]] = None,
    mean_return: float = 0.08,
    std_dev: float = 0.16,
    random_state: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate the evolution of a portfolio over multiple years.

    This function follows the steps outlined in the specification.  It
    returns a tidy DataFrame with one row per year summarising the
    portfolio after dividend and rebalancing.

    Args:
        asset_alloc_df: Initial asset allocation table.  Must include
            columns ``Item``, ``Allocation``, ``Beta``, ``Monthly Liquidity %``
            and ``Private %``.
        liquidity_df: Liquidity waterfall table with columns ``Item`` and
            ``Liquidity Order``.
        cash_flows_df: Cash flow projections with columns ``Item``,
            ``Projection Year``, ``Capital Call %``, ``Distribution %``.
        baseline_return: Baseline equity return used in the private cash
            flow formula (e.g. 0.08 for 8 %).  Default 0.08.
        baseline_std: Baseline standard deviation (unused directly but
            provided for completeness).  Default 0.16.
        annual_dividend: Annual dividend amount.  If ``dividend_is_percent``
            is False, the value is treated as a fixed dollar amount.  If
            True, it is interpreted as a fraction of the pre‑dividend
            portfolio NAV.
        dividend_is_percent: Whether ``annual_dividend`` is a percentage of
            the portfolio NAV rather than a fixed dollar amount.
        n_years: Number of projection years, including year 1 shock.
        scenario_returns: Optional mapping of year index (1‑based) to a
            predetermined market return (e.g. {2: 0.3, 3: 0.3}).  Years
            absent from the mapping are drawn from a normal distribution
            using ``mean_return`` and ``std_dev``.
        mean_return: Mean of the normal distribution used for random
            returns when ``scenario_returns`` does not specify a value.
        std_dev: Standard deviation of the normal distribution used for
            random returns.
        random_state: Optional numpy random number generator.  If
            ``None``, ``np.random.default_rng()`` is used.

    Returns:
        DataFrame with one row per year containing: ``year``,
        ``nav_total``, ``beta_total``, ``private_total``,
        ``monthly_liquid_total``, ``market_return``, ``navs`` (a
        dictionary mapping each item to its NAV).  The ``navs`` column
        can be expanded or used as needed in the user interface.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    # Initialize line items
    items = _initialize_line_items(asset_alloc_df)
    # Determine the cash bucket name (first with beta==0)
    cash_item_name = next((name for name, li in items.items() if abs(li.beta) < 1e-6), 'Cash')
    # Liquidity order
    liquidity_order = _initialize_liquidity_order(liquidity_df, items)
    # Cash flow projections
    cash_flows = _initialize_cash_flows(cash_flows_df)
    # Compute starting metrics
    total_nav, beta_start, private_start = _compute_portfolio_metrics(items)
    # Preallocate list of results
    results: List[Dict[str, object]] = []
    # Market return for year 1 is the shock: –40 %
    market_returns: Dict[int, float] = {1: -0.40}
    # Add scenario returns override
    if scenario_returns:
        for k, v in scenario_returns.items():
            market_returns[k] = v
    private_cash_flow_items = set(cash_flows.keys())
    # Simulate each year
    for year in range(1, n_years + 1):
        # Determine market return for this year
        if year in market_returns:
            mr = float(market_returns[year])
        else:
            mr = float(rng.normal(loc=mean_return, scale=std_dev))
        # Apply returns in two branches:
        # 1) items in cash flow table -> private cash-flow logic
        # 2) all other items -> public return logic
        _apply_public_returns(
            items,
            mr,
            cash_item_name,
            private_cash_flow_items=private_cash_flow_items,
        )
        _apply_private_cash_flows(
            items,
            cash_flows,
            year,
            baseline_return,
            mr,
            cash_item_name,
        )
        # Compute metrics before dividend
        total_nav_pre, beta_pre, private_pre = _compute_portfolio_metrics(items)
        monthly_liq_pre = sum(li.nav * li.monthly_liq for li in items.values()) / total_nav_pre if total_nav_pre > 0 else 0.0
        # Determine dividend amount for this year
        if annual_dividend > 0:
            if dividend_is_percent:
                dividend_amt = annual_dividend * total_nav_pre
            else:
                dividend_amt = annual_dividend
        else:
            dividend_amt = 0.0
        # Apply dividend using liquidity waterfall
        _apply_dividend(items, liquidity_order, dividend_amt)
        # Compute metrics after dividend, before rebalancing
        total_nav_post, beta_post, private_post = _compute_portfolio_metrics(items)
        monthly_liq_post = sum(li.nav * li.monthly_liq for li in items.values()) / total_nav_post if total_nav_post > 0 else 0.0
        # Rebalance if beta deviates
        _rebalance_portfolio(items, beta_start, tolerance=0.02)
        # Compute metrics after rebalancing
        total_nav_rb, beta_rb, private_rb = _compute_portfolio_metrics(items)
        monthly_liq_rb = sum(li.nav * li.monthly_liq for li in items.values()) / total_nav_rb if total_nav_rb > 0 else 0.0
        # Record results
        # Capture per‑item metrics (NAV, beta, monthly liquidity fraction, private fraction)
        item_metrics = {
            name: {
                'nav': li.nav,
                'beta': li.beta,
                'monthly_liq': li.monthly_liq,
                'private': li.private,
            }
            for name, li in items.items()
        }
        results.append({
            'year': year,
            'market_return': mr,
            'nav_total_pre': total_nav_pre,
            'beta_pre': beta_pre,
            'private_pre': private_pre,
            'monthly_liq_pre': monthly_liq_pre,
            'nav_total_post': total_nav_post,
            'beta_post': beta_post,
            'private_post': private_post,
            'monthly_liq_post': monthly_liq_post,
            'nav_total': total_nav_rb,
            'beta_total': beta_rb,
            'private_total': private_rb,
            'monthly_liq_total': monthly_liq_rb,
            'items': item_metrics,
        })
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


def run_multiple_simulations(
    n_paths: int,
    asset_alloc_df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    cash_flows_df: pd.DataFrame,
    baseline_return: float = 0.08,
    baseline_std: float = 0.16,
    annual_dividend: float = 0.0,
    dividend_is_percent: bool = False,
    n_years: int = 10,
    mean_return: float = 0.08,
    std_dev: float = 0.16,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run multiple simulation paths and return aggregated results.

    Args:
        n_paths: Number of Monte Carlo paths to simulate.
        asset_alloc_df: Initial asset allocation DataFrame.
        liquidity_df: Liquidity waterfall DataFrame.
        cash_flows_df: Cash flow projection DataFrame.
        baseline_return: Baseline return for private flow adjustments.
        baseline_std: Baseline standard deviation (unused directly).
        annual_dividend: Dividend amount or fraction.
        dividend_is_percent: Whether the dividend is a percentage of NAV.
        n_years: Number of projection years (including year 1 shock).
        mean_return: Mean of normal distribution for random returns.
        std_dev: Standard deviation of normal distribution for random returns.
        random_seed: Optional seed for reproducible simulations.

    Returns:
        DataFrame with columns ``path``, ``year``, ``nav_total``,
        ``beta_total``, ``private_total`` and ``monthly_liq_total``.  Each
        row represents the end‐of‐year metrics (after dividend and
        rebalancing) for a given path and year.
    """
    rng = np.random.default_rng(random_seed)
    records: List[Dict[str, object]] = []
    for path in range(n_paths):
        # For each path, use a new random state derived from the master RNG
        # to ensure independence
        path_seed = rng.integers(0, 10**9)
        df = simulate_portfolio(
            asset_alloc_df=asset_alloc_df,
            liquidity_df=liquidity_df,
            cash_flows_df=cash_flows_df,
            baseline_return=baseline_return,
            baseline_std=baseline_std,
            annual_dividend=annual_dividend,
            dividend_is_percent=dividend_is_percent,
            n_years=n_years,
            scenario_returns=None,
            mean_return=mean_return,
            std_dev=std_dev,
            random_state=np.random.default_rng(int(path_seed)),
        )
        for _, row in df.iterrows():
            records.append({
                'path': path,
                'year': int(row['year']),
                'nav_total': row['nav_total'],
                'beta_total': row['beta_total'],
                'private_total': row['private_total'],
                'monthly_liq_total': row['monthly_liq_total'],
                'market_return': row['market_return'],
            })
    return pd.DataFrame(records)