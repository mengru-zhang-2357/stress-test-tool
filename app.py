"""
app.py
------

This module defines a Shiny application that wraps the stress testing
simulation defined in ``stress_simulation.py``.  The user can input an
initial portfolio, liquidity preferences, private asset cash flows, an
annual dividend, and baseline return assumptions.  The app produces
deterministic scenario analyses (V‑shaped and U‑shaped recovery paths)
and a Monte Carlo simulation with a user‑specified number of random
paths.  Results are visualised using Plotly charts.

The Shiny interface uses editable ``DataGrid`` widgets for data entry.
Cell edits propagate back to the server via the reactive
``.data_view()`` method of the data frame outputs【925037061954615†L389-L439】.  To
enable cell editing, ``editable=True`` is passed to ``render.DataGrid``【925037061954615†L296-L333】.

"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive
from typing import Tuple, Iterable

from stress_simulation import simulate_portfolio, run_multiple_simulations

# Import shinyswatch to provide Bootswatch themes.  When applying a theme,
# pass the theme object via the `theme` argument of the top-level page
# container.  In this app, we use the Sandstone Bootswatch theme.
import shinyswatch


# ----------------------------------------------------------------------
# Default input tables.  These provide a starting point for the user to
# modify.  The user can edit any cell in the data grids.  The tables
# should contain the columns specified in the problem statement.
# ----------------------------------------------------------------------

default_asset_alloc = pd.DataFrame({
    'Item': ['Public Equity', 'Hedge Funds', 'Buyout', 'Venture Capital', 'Real Estate', 'Natural Resources', 'Cash'],
    'Allocation': [300.0, 150.0, 150.0, 200.0, 50.0, 50.0, 100.0],
    'Beta': [0.95, 0.4, 1.0, 1.0, 0.6, 0.6, 0.0],
    'Monthly Liquidity %': [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0],
    'Private %': [0.2, 0.05, 1.0, 1.0, 1.0, 1.0, 0.0],
})

default_liquidity = pd.DataFrame({
    'Item': ['Cash', 'Public Equity', 'Hedge Funds'],
    'Liquidity Order': [1, 2, 3],
})

asset_classes = ["Buyout", "Venture Capital", "Real Estate", "Natural Resources"]

default_cash_flows = pd.DataFrame({
    "Item": sum([[asset] * 5 for asset in asset_classes], []),
    "Projection Year": [1, 2, 3, 4, 5] * len(asset_classes),
    "Capital Call %": [0.10, 0.10, 0.10, 0.10, 0.10] * len(asset_classes),
    "Distribution %": [0.10, 0.10, 0.10, 0.10, 0.10] * len(asset_classes),
})


def build_app_ui() -> ui.NavbarPage:
    """Construct the Shiny user interface."""
    # When using ui.page_navbar, pass the navigation panels as positional arguments
    # and the title as a named keyword argument (title=...).  This avoids errors
    # where the first argument is interpreted as a navigation item instead of the title.
    return ui.page_navbar(
        ui.tags.style("""
            .row-action-btn { width: 220px; margin-top: 0.4rem; margin-bottom: 0.4rem; }
            .result-chart { margin-bottom: 1rem; }
        """),
        # First tab: inputs
        ui.nav_panel(
            "Inputs",
            ui.layout_sidebar(
                # Sidebar is provided as the first argument. The following arguments define the main content.
                ui.sidebar(
                    ui.h4("Parameters"),
                    ui.input_numeric("dividend_amount", "Annual dividend (absolute or fraction)", 0.0),
                    ui.input_radio_buttons(
                        "dividend_type",
                        "Dividend type",
                        choices=["Dollar", "Percent"],
                        selected="Dollar",
                    ),
                    ui.input_numeric(
                        "baseline_return",
                        "Baseline equity return (decimal)",
                        0.08,
                    ),
                    ui.input_numeric(
                        "baseline_std",
                        "Baseline equity volatility (decimal)",
                        0.16,
                    ),
                    ui.input_numeric(
                        "n_years",
                        "Projection horizon (years)",
                        5,
                    ),
                    ui.hr(),
                    ui.h5("Monte Carlo settings"),
                    ui.input_numeric(
                        "n_paths",
                        "Number of simulation paths",
                        100,
                    ),
                    ui.input_action_button("simulate", "Simulate"),
                ),
                # Main content: outputs and descriptions
                ui.h3("Starting asset allocation"),
                ui.p(
                    "Enter the starting allocation for each line item.  "
                    "Allocation amounts are in dollars.  Monthly liquidity and private "
                    "percentages can be specified either as decimals (e.g. 0.8) or percentages (e.g. 80)."
                ),
                ui.output_data_frame("asset_alloc_table"),
                # Buttons to add or delete rows in the asset allocation table
                ui.input_action_button("add_asset_row", "Add row", class_="row-action-btn"),
                ui.input_action_button("delete_asset_row", "Delete selected row(s)", class_="btn-danger row-action-btn"),
                ui.br(),
                ui.h3("Liquidity waterfall"),
                ui.p(
                    "Specify the order in which assets are tapped for liquidity.  "
                    "Lower numbers mean higher priority.  Items omitted from this table "
                    "are appended to the end of the waterfall."
                ),
                ui.output_data_frame("liquidity_table"),
                ui.input_action_button("add_liq_row", "Add row", class_="row-action-btn"),
                ui.input_action_button("delete_liq_row", "Delete selected row(s)", class_="btn-danger row-action-btn"),
                ui.br(),
                ui.h3("Private asset cash flows"),
                ui.p(
                    "For each private asset class, specify capital call and distribution "
                    "percentages by projection year.  Percentages can be decimals or percent values."
                ),
                ui.output_data_frame("cash_flow_table"),
                ui.input_action_button("add_cf_row", "Add row", class_="row-action-btn"),
                ui.input_action_button("delete_cf_row", "Delete selected row(s)", class_="btn-danger row-action-btn"),
            ),
        ),
        # Second tab: results
        ui.nav_panel(
            "Results",
            ui.h3("Scenario outputs"),
            ui.accordion(
                ui.accordion_panel(
                    "V-shaped",
                    ui.row(
                        ui.column(6, ui.div(ui.output_plot("v_market_plot"), class_="result-chart")),
                        ui.column(6, ui.div(ui.output_plot("v_nav_plot"), class_="result-chart")),
                    ),
                    ui.row(
                        ui.column(6, ui.div(ui.output_plot("v_beta_plot"), class_="result-chart")),
                        ui.column(6, ui.div(ui.output_plot("v_private_plot"), class_="result-chart")),
                    ),
                    ui.input_action_button("show_v_table", "Show underlying data", class_="btn-outline-secondary"),
                ),
                ui.accordion_panel(
                    "U-shaped",
                    ui.row(
                        ui.column(6, ui.div(ui.output_plot("u_market_plot"), class_="result-chart")),
                        ui.column(6, ui.div(ui.output_plot("u_nav_plot"), class_="result-chart")),
                    ),
                    ui.row(
                        ui.column(6, ui.div(ui.output_plot("u_beta_plot"), class_="result-chart")),
                        ui.column(6, ui.div(ui.output_plot("u_private_plot"), class_="result-chart")),
                    ),
                    ui.input_action_button("show_u_table", "Show underlying data", class_="btn-outline-secondary"),
                ),
                ui.accordion_panel(
                    "Monte Carlo",
                    ui.row(
                        ui.column(6, ui.div(ui.output_plot("mc_market_plot"), class_="result-chart")),
                        ui.column(6, ui.div(ui.output_plot("nav_fan_plot"), class_="result-chart")),
                    ),
                    ui.row(
                        ui.column(6, ui.div(ui.output_plot("beta_fan_plot"), class_="result-chart")),
                        ui.column(6, ui.div(ui.output_plot("private_fan_plot"), class_="result-chart")),
                    ),
                ),
            ),
        ),
        title="Portfolio Stress Test",
        # Apply the Sandstone Bootswatch theme via shinyswatch.  The theme must
        # be passed to the `theme` argument of the page function (see
        # https://posit-dev.github.io/py-shinyswatch/reference/theme.sandstone.html)
        theme=shinyswatch.theme.sandstone,
    )


def build_server():
    """Construct the Shiny server logic."""
    def server(input, output, session):
        # Reactive values to hold the current contents of the input tables.
        # Using reactive.value allows us to update the data frames in response to
        # 'Add row' button clicks.  When these values change, the data grids will
        # re-render automatically.
        asset_df_val = reactive.value(default_asset_alloc.copy())
        liq_df_val = reactive.value(default_liquidity.copy())
        cf_df_val = reactive.value(default_cash_flows.copy())

        # --- Data tables: asset allocation ---
        @render.data_frame
        def asset_alloc_table():
            # Provide editable data grid for asset allocation.
            # Use selection_mode="rows" to enable row selection/editing when editable=True.
            return render.DataGrid(
                asset_df_val().copy(),
                editable=True,
                selection_mode="rows",
            )

        @render.data_frame
        def liquidity_table():
            return render.DataGrid(
                liq_df_val().copy(),
                editable=True,
                selection_mode="rows",
            )

        @render.data_frame
        def cash_flow_table():
            return render.DataGrid(
                cf_df_val().copy(),
                editable=True,
                selection_mode="rows",
            )

        # Handle adding new rows to each table when the user clicks the
        # corresponding Add row button.  These functions update the reactive
        # values, which in turn triggers re-rendering of the grids.
        # When the user clicks the "Add row" button for the asset allocation table,
        # append a blank row.  The @reactive.event decorator should be stacked
        # below @reactive.effect so that it is applied first (see
        # https://shiny.posit.co/py/api/core/reactive.event.html).  Placing
        # @reactive.event closer to the function ensures the function runs only
        # when the event triggers.
        @reactive.effect
        @reactive.event(input.add_asset_row)
        def add_asset_row_event():
            """Append a blank row to the asset allocation table.

            Before appending, capture the current patched data using
            ``asset_alloc_table.data_view()`` so that any unsaved cell edits
            are included.  After adding the new row, update the reactive
            value and refresh the DataGrid via ``update_data``.
            """
            # Use data_view to include cell patches
            df = asset_alloc_table.data_view().reset_index(drop=True)
            # Append a default row
            new_row = {
                'Item': '',
                'Allocation': 0.0,
                'Beta': 0.0,
                'Monthly Liquidity %': 0.0,
                'Private %': 0.0,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            asset_df_val.set(df)

        # When the user clicks the "Add row" button for the liquidity table,
        # append a new row.  @reactive.event goes below @reactive.effect so
        # that it is applied first.
        @reactive.effect
        @reactive.event(input.add_liq_row)
        def add_liq_row_event():
            """Append a blank row to the liquidity waterfall table.

            Use ``data_view`` to capture unsaved edits.  Automatically assign the
            next integer Liquidity Order by inspecting the current values.
            """
            df = liquidity_table.data_view().reset_index(drop=True)
            # Determine next order (max + 1 or 1 if none)
            if 'Liquidity Order' in df.columns and not df.empty:
                # Convert to numeric and ignore errors
                try:
                    next_order = int(pd.to_numeric(df['Liquidity Order'], errors='coerce').max()) + 1
                except Exception:
                    next_order = len(df) + 1
            else:
                next_order = 1
            new_row = {
                'Item': '',
                'Liquidity Order': next_order,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            liq_df_val.set(df)

        # When the user clicks the "Add row" button for the cash flow table,
        # append a new row.  Stack @reactive.event below @reactive.effect to
        # apply it first.
        @reactive.effect
        @reactive.event(input.add_cf_row)
        def add_cf_row_event():
            """Append a blank row to the cash flow table.

            The current patched data is obtained via ``cash_flow_table.data_view()``.
            A new projection year is chosen by taking the maximum existing year
            (converted to numeric) and adding one.  This prevents the table from
            resetting when rows are added.
            """
            df = cash_flow_table.data_view().reset_index(drop=True)
            # Determine next projection year
            if 'Projection Year' in df.columns and not df.empty:
                try:
                    next_year = int(pd.to_numeric(df['Projection Year'], errors='coerce').max()) + 1
                except Exception:
                    next_year = len(df) + 1
            else:
                next_year = 1
            new_row = {
                'Item': '',
                'Projection Year': next_year,
                'Capital Call %': 0.0,
                'Distribution %': 0.0,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            cf_df_val.set(df)

        # Delete selected rows from asset allocation
        @reactive.effect
        @reactive.event(input.delete_asset_row)
        def delete_asset_row_event():
            """Delete the selected rows from the asset allocation table.

            Uses the DataGrid's cell selection to determine which rows to remove.  If
            no rows are selected, nothing happens.  After deletion, update the
            reactive value and refresh the DataGrid.
            """
            # Get current patched data to capture user edits
            df = asset_alloc_table.data_view().reset_index(drop=True)
            # Determine selected rows via DataGrid cell selection
            sel = asset_alloc_table.input_cell_selection()
            rows_to_drop = []
            if sel and isinstance(sel, dict):
                rows_to_drop = list(sel.get('rows', []))
            if rows_to_drop:
                df = df.drop(index=rows_to_drop).reset_index(drop=True)
                asset_df_val.set(df)

        # Delete selected rows from liquidity waterfall table
        @reactive.effect
        @reactive.event(input.delete_liq_row)
        def delete_liq_row_event():
            df = liquidity_table.data_view().reset_index(drop=True)
            sel = liquidity_table.input_cell_selection()
            rows_to_drop = []
            if sel and isinstance(sel, dict):
                rows_to_drop = list(sel.get('rows', []))
            if rows_to_drop:
                df = df.drop(index=rows_to_drop).reset_index(drop=True)
                liq_df_val.set(df)

        # Delete selected rows from cash flow table
        @reactive.effect
        @reactive.event(input.delete_cf_row)
        def delete_cf_row_event():
            df = cash_flow_table.data_view().reset_index(drop=True)
            sel = cash_flow_table.input_cell_selection()
            rows_to_drop = []
            if sel and isinstance(sel, dict):
                rows_to_drop = list(sel.get('rows', []))
            if rows_to_drop:
                df = df.drop(index=rows_to_drop).reset_index(drop=True)
                cf_df_val.set(df)

        # Helper: parse user inputs into data frames
        def get_user_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            """Return the current versions of the input tables with numeric columns coerced.

            This helper uses `.data_view()` on each data frame output to pick up any
            pending cell edits.  It then converts numeric columns to floats/ints
            using ``pd.to_numeric`` with ``errors='coerce'`` and fills missing
            values with sensible defaults (zero).  This ensures that the
            simulation functions do not encounter string concatenation errors or
            type mismatches when performing arithmetic.
            """
            df_asset: pd.DataFrame = asset_alloc_table.data_view().reset_index(drop=True)
            df_liq: pd.DataFrame = liquidity_table.data_view().reset_index(drop=True)
            df_cf: pd.DataFrame = cash_flow_table.data_view().reset_index(drop=True)
            # Coerce numeric columns in asset allocation
            if not df_asset.empty:
                num_cols = ['Allocation', 'Beta', 'Monthly Liquidity %', 'Private %']
                for col in num_cols:
                    if col in df_asset.columns:
                        df_asset[col] = pd.to_numeric(df_asset[col], errors='coerce').fillna(0.0)
                # Strip item names (ensure string type)
                df_asset['Item'] = df_asset['Item'].astype(str)
            # Coerce numeric columns in liquidity table
            if not df_liq.empty:
                if 'Liquidity Order' in df_liq.columns:
                    df_liq['Liquidity Order'] = pd.to_numeric(df_liq['Liquidity Order'], errors='coerce').fillna(0).astype(int)
                df_liq['Item'] = df_liq['Item'].astype(str)
            # Coerce numeric columns in cash flow table
            if not df_cf.empty:
                for col in ['Projection Year', 'Capital Call %', 'Distribution %']:
                    if col in df_cf.columns:
                        df_cf[col] = pd.to_numeric(df_cf[col], errors='coerce').fillna(0.0)
                df_cf['Item'] = df_cf['Item'].astype(str)
            return df_asset, df_liq, df_cf

        def normalize_dividend_amount(div_amt: float, dividend_is_percent: bool) -> float:
            """Normalize percent-style dividend inputs to decimal fractions."""
            if dividend_is_percent and abs(div_amt) > 1.0:
                return div_amt / 100.0
            return div_amt

        # Scenario 1: V‑shaped recovery
        # Use reactive.calc instead of reactive.memoize (memoize is removed in recent Shiny versions).
        @reactive.calc
        def v_scenario_results():
            df_asset, df_liq, df_cf = get_user_tables()
            years = int(input.n_years())
            # Use baseline return and volatility from inputs
            br = float(input.baseline_return())
            bstd = float(input.baseline_std())
            div_amt = float(input.dividend_amount())
            div_type = str(input.dividend_type())
            dividend_is_percent = div_type.lower().startswith("p")
            div_amt = normalize_dividend_amount(div_amt, dividend_is_percent)
            # Define predetermined returns: years 2 and 3 at +30 %, remainder normal
            scenario = {2: 0.30, 3: 0.30}
            df = simulate_portfolio(
                asset_alloc_df=df_asset,
                liquidity_df=df_liq,
                cash_flows_df=df_cf,
                baseline_return=br,
                baseline_std=bstd,
                annual_dividend=div_amt,
                dividend_is_percent=dividend_is_percent,
                n_years=years,
                scenario_returns=scenario,
                mean_return=br,
                std_dev=bstd,
            )
            # Insert a baseline row (Year 0) using starting allocations as reference.
            # Compute starting NAV, weighted beta, and weighted private fraction.
            nav0 = df_asset['Allocation'].sum()
            # Helper to convert percentage-like values to fractions (0–1)
            def _to_fraction_local(val):
                try:
                    val_float = float(val)
                except Exception:
                    return 0.0
                if pd.isna(val_float):
                    return 0.0
                return val_float / 100.0 if abs(val_float) > 1.0 else val_float
            if nav0 > 0:
                beta0 = (df_asset['Allocation'] * df_asset['Beta'].astype(float)).sum() / nav0
                private_frac_series = df_asset['Private %'].apply(_to_fraction_local)
                private0 = (df_asset['Allocation'] * private_frac_series).sum() / nav0
            else:
                beta0 = 0.0
                private0 = 0.0
            baseline_row = pd.DataFrame([
                {
                    'year': 0,
                    'nav_total_pre': nav0,
                    'nav_total_post': nav0,
                    'nav_total': nav0,
                    'beta_pre': beta0,
                    'beta_post': beta0,
                    'beta_total': beta0,
                    'private_pre': private0,
                    'private_post': private0,
                    'private_total': private0,
                    'market_return': 0.0,
                }
            ])
            df = pd.concat([baseline_row, df], ignore_index=True)
            return df

        @reactive.calc
        def u_scenario_results():
            df_asset, df_liq, df_cf = get_user_tables()
            years = int(input.n_years())
            br = float(input.baseline_return())
            bstd = float(input.baseline_std())
            div_amt = float(input.dividend_amount())
            div_type = str(input.dividend_type())
            dividend_is_percent = div_type.lower().startswith("p")
            div_amt = normalize_dividend_amount(div_amt, dividend_is_percent)
            # U‑shaped recovery: years 2–6 gradually recover (8% to 12%)
            scenario = {}
            if years >= 2:
                scenario[2] = 0.08
            if years >= 3:
                scenario[3] = 0.09
            if years >= 4:
                scenario[4] = 0.10
            if years >= 5:
                scenario[5] = 0.11
            if years >= 6:
                scenario[6] = 0.12
            df = simulate_portfolio(
                asset_alloc_df=df_asset,
                liquidity_df=df_liq,
                cash_flows_df=df_cf,
                baseline_return=br,
                baseline_std=bstd,
                annual_dividend=div_amt,
                dividend_is_percent=dividend_is_percent,
                n_years=years,
                scenario_returns=scenario,
                mean_return=br,
                std_dev=bstd,
            )
            # Insert a baseline row (Year 0) using starting allocations as reference.
            nav0 = df_asset['Allocation'].sum()
            def _to_fraction_local(val):
                try:
                    val_float = float(val)
                except Exception:
                    return 0.0
                if pd.isna(val_float):
                    return 0.0
                return val_float / 100.0 if abs(val_float) > 1.0 else val_float
            if nav0 > 0:
                beta0 = (df_asset['Allocation'] * df_asset['Beta'].astype(float)).sum() / nav0
                private_frac_series = df_asset['Private %'].apply(_to_fraction_local)
                private0 = (df_asset['Allocation'] * private_frac_series).sum() / nav0
            else:
                beta0 = 0.0
                private0 = 0.0
            baseline_row = pd.DataFrame([
                {
                    'year': 0,
                    'nav_total_pre': nav0,
                    'nav_total_post': nav0,
                    'nav_total': nav0,
                    'beta_pre': beta0,
                    'beta_post': beta0,
                    'beta_total': beta0,
                    'private_pre': private0,
                    'private_post': private0,
                    'private_total': private0,
                    'market_return': 0.0,
                }
            ])
            df = pd.concat([baseline_row, df], ignore_index=True)
            return df

        def format_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
            display_names = {
                'year': 'Year',
                'nav_total_pre': 'NAV before dividend',
                'nav_total_post': 'NAV after dividend',
                'nav_total': 'NAV after rebalance',
                'beta_pre': 'Beta before dividend',
                'beta_post': 'Beta after dividend',
                'beta_total': 'Beta after rebalance',
                'private_pre': 'Private % before dividend',
                'private_post': 'Private % after dividend',
                'private_total': 'Private % after rebalance',
                'market_return': 'Market return',
            }
            table = df[list(display_names.keys())].copy().rename(columns=display_names)
            nav_cols = ['NAV before dividend', 'NAV after dividend', 'NAV after rebalance']
            for col in nav_cols:
                table[col] = table[col].map(lambda x: f"${float(x):,.2f}")
            pct_cols = ['Private % before dividend', 'Private % after dividend', 'Private % after rebalance', 'Market return']
            for col in pct_cols:
                table[col] = table[col].map(lambda x: f"{float(x) * 100:.2f}%")
            for col in ['Beta before dividend', 'Beta after dividend', 'Beta after rebalance']:
                table[col] = table[col].map(lambda x: round(float(x), 3))
            table['Year'] = table['Year'].astype(int)
            return table

        def show_table_modal(title: str, table_df: pd.DataFrame):
            ui.modal_show(
                ui.modal(
                    ui.p("Column guide: pre = after return/cash-flow and before dividend; post = after dividend; after rebalance = final year-end value."),
                    ui.output_data_frame("scenario_modal_table"),
                    title=title,
                    size="l",
                    easy_close=True,
                    footer=ui.modal_button("Close"),
                )
            )
            modal_table_val.set(table_df)

        modal_table_val = reactive.value(pd.DataFrame())

        # Render scenario tables and plots
        @render.data_frame
        def scenario_modal_table():
            df = modal_table_val()
            if df is None or df.empty:
                return render.DataTable(pd.DataFrame())
            return render.DataTable(df, filters=True)

        @reactive.effect
        @reactive.event(input.show_v_table)
        def _show_v_table_modal():
            show_table_modal("V-shaped scenario data", format_scenario_table(v_scenario_results()))

        @reactive.effect
        @reactive.event(input.show_u_table)
        def _show_u_table_modal():
            show_table_modal("U-shaped scenario data", format_scenario_table(u_scenario_results()))

        @render.data_frame
        def v_scenario_table():
            df = v_scenario_results()
            # Show aggregated metrics by year: total NAV, beta, private
            table = df[[
                'year', 'nav_total_pre', 'nav_total_post', 'nav_total',
                'beta_pre', 'beta_post', 'beta_total',
                'private_pre', 'private_post', 'private_total',
                'market_return',
            ]].copy()
            # Round numeric columns to two decimal places for readability, but keep
            # the year column as integer.  Convert floats explicitly to avoid
            # rounding errors.
            years = table['year'].astype(int)
            for col in table.columns:
                if col != 'year':
                    table[col] = table[col].astype(float).round(2)
            table['year'] = years
            return render.DataTable(table, filters=True)

        @render.plot(alt="Market growth plot (V-shaped)")
        def v_market_plot():
            df = v_scenario_results()
            # Create a matplotlib figure for the market growth of $1
            fig, ax = plt.subplots()
            if df.empty:
                # Empty plot if no data
                ax.plot([])
            else:
                growth = [1.0]
                scenario_df = df[df['year'] > 0]
                for mr in scenario_df['market_return']:
                    growth.append(growth[-1] * (1.0 + mr))
                years = [0] + scenario_df['year'].astype(int).tolist()
                ax.plot(years, growth, marker='o')
            ax.set_xlabel('Year')
            ax.set_ylabel('Growth of $1')
            ax.set_title('Market index growth (V-shaped)')
            return fig

        @render.data_frame
        def u_scenario_table():
            df = u_scenario_results()
            table = df[[
                'year', 'nav_total_pre', 'nav_total_post', 'nav_total',
                'beta_pre', 'beta_post', 'beta_total',
                'private_pre', 'private_post', 'private_total',
                'market_return',
            ]].copy()
            years = table['year'].astype(int)
            for col in table.columns:
                if col != 'year':
                    table[col] = table[col].astype(float).round(2)
            table['year'] = years
            return render.DataTable(table, filters=True)

        @render.plot(alt="Market growth plot (U-shaped)")
        def u_market_plot():
            df = u_scenario_results()
            fig, ax = plt.subplots()
            if df.empty:
                ax.plot([])
            else:
                growth = [1.0]
                scenario_df = df[df['year'] > 0]
                for mr in scenario_df['market_return']:
                    growth.append(growth[-1] * (1.0 + mr))
                years = [0] + scenario_df['year'].astype(int).tolist()
                ax.plot(years, growth, marker='o')
            ax.set_xlabel('Year')
            ax.set_ylabel('Growth of $1')
            ax.set_title('Market index growth (U-shaped)')
            return fig

        @render.plot(alt="V-shaped NAV path")
        def v_nav_plot():
            df = v_scenario_results()
            fig, ax = plt.subplots()
            ax.plot(df['year'], df['nav_total'], marker='o')
            ax.set_title('NAV total path (V-shaped)')
            ax.set_xlabel('Year')
            ax.set_ylabel('NAV')
            return fig

        @render.plot(alt="V-shaped beta path")
        def v_beta_plot():
            df = v_scenario_results()
            fig, ax = plt.subplots()
            ax.plot(df['year'], df['beta_total'], marker='o')
            ax.set_title('Beta total path (V-shaped)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Beta')
            return fig

        @render.plot(alt="V-shaped private path")
        def v_private_plot():
            df = v_scenario_results()
            fig, ax = plt.subplots()
            ax.plot(df['year'], df['private_total'], marker='o')
            ax.set_title('Private total path (V-shaped)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Private %')
            return fig

        @render.plot(alt="U-shaped NAV path")
        def u_nav_plot():
            df = u_scenario_results()
            fig, ax = plt.subplots()
            ax.plot(df['year'], df['nav_total'], marker='o')
            ax.set_title('NAV total path (U-shaped)')
            ax.set_xlabel('Year')
            ax.set_ylabel('NAV')
            return fig

        @render.plot(alt="U-shaped beta path")
        def u_beta_plot():
            df = u_scenario_results()
            fig, ax = plt.subplots()
            ax.plot(df['year'], df['beta_total'], marker='o')
            ax.set_title('Beta total path (U-shaped)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Beta')
            return fig

        @render.plot(alt="U-shaped private path")
        def u_private_plot():
            df = u_scenario_results()
            fig, ax = plt.subplots()
            ax.plot(df['year'], df['private_total'], marker='o')
            ax.set_title('Private total path (U-shaped)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Private %')
            return fig

        # Monte Carlo simulation event reactive
        @reactive.event(input.simulate)
        def mc_results():
            df_asset, df_liq, df_cf = get_user_tables()
            years = int(input.n_years())
            br = float(input.baseline_return())
            bstd = float(input.baseline_std())
            div_amt = float(input.dividend_amount())
            div_type = str(input.dividend_type())
            dividend_is_percent = div_type.lower().startswith("p")
            div_amt = normalize_dividend_amount(div_amt, dividend_is_percent)
            n_paths = int(input.n_paths())
            # Run simulation
            res = run_multiple_simulations(
                n_paths=n_paths,
                asset_alloc_df=df_asset,
                liquidity_df=df_liq,
                cash_flows_df=df_cf,
                baseline_return=br,
                baseline_std=bstd,
                annual_dividend=div_amt,
                dividend_is_percent=dividend_is_percent,
                n_years=years,
                mean_return=br,
                std_dev=bstd,
                random_seed=None,
            )
            return res

        # Helper to compute quantile tables
        def compute_quantiles(df: pd.DataFrame, column: str, quantiles: Iterable[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
            if df.empty or column not in df:
                return pd.DataFrame()
            q_df = df.groupby('year')[column].quantile(list(quantiles)).unstack()
            q_df.columns = [f'{int(q*100)}th' for q in quantiles]
            q_df.index.name = 'Year'
            q_df.reset_index(inplace=True)
            # Round the numeric columns (quantile values) to two decimals
            if not q_df.empty:
                for col in q_df.columns:
                    if col != 'Year':
                        q_df[col] = q_df[col].astype(float).round(2)
                q_df['Year'] = q_df['Year'].astype(int)
            return q_df

        # Render fan charts
        @render.plot(alt="Market index growth fan chart")
        def mc_market_plot():
            try:
                df = mc_results()
            except Exception:
                df = None
            fig, ax = plt.subplots()
            if df is None or df.empty or 'market_return' not in df.columns:
                ax.plot([])
                ax.set_title('Market index growth fan chart')
                ax.set_xlabel('Year')
                ax.set_ylabel('Growth of $1')
                return fig
            growth_df = df[['path', 'year', 'market_return']].copy()
            growth_df = growth_df.sort_values(['path', 'year'])
            growth_df['growth'] = growth_df.groupby('path')['market_return'].transform(lambda s: (1 + s).cumprod())
            q_df = compute_quantiles(growth_df.rename(columns={'growth': 'market_growth'}), 'market_growth')
            if not q_df.empty:
                baseline_row = {'Year': 0}
                for col in q_df.columns[1:]:
                    baseline_row[col] = 1.0
                q_df = pd.concat([pd.DataFrame([baseline_row]), q_df], ignore_index=True)
                for col in q_df.columns[1:]:
                    ax.plot(q_df['Year'], q_df[col], label=col)
                ax.legend()
            ax.set_title('Market index growth fan chart')
            ax.set_xlabel('Year')
            ax.set_ylabel('Growth of $1')
            return fig

        @render.plot(alt="Portfolio NAV fan chart")
        def nav_fan_plot():
            # Return a Matplotlib figure for the NAV fan chart
            try:
                df = mc_results()
            except Exception:
                df = None
            if df is None or df.empty:
                fig, ax = plt.subplots()
                ax.plot([])
                ax.set_title('Portfolio NAV fan chart')
                ax.set_xlabel('Year')
                ax.set_ylabel('NAV')
                return fig
            q_df = compute_quantiles(df, 'nav_total')
            # Insert a baseline row (Year 0) corresponding to the starting portfolio NAV.
            # Compute the baseline using the current asset allocation table.
            df_asset, _, _ = get_user_tables()
            nav0 = df_asset['Allocation'].sum()
            if not q_df.empty:
                baseline_row = {'Year': 0}
                for col in q_df.columns[1:]:
                    baseline_row[col] = round(nav0, 2)
                q_df = pd.concat([pd.DataFrame([baseline_row]), q_df], ignore_index=True)
            fig, ax = plt.subplots()
            for col in q_df.columns[1:]:
                ax.plot(q_df['Year'], q_df[col], label=col)
            ax.set_title('Portfolio NAV fan chart')
            ax.set_xlabel('Year')
            ax.set_ylabel('NAV')
            ax.legend()
            return fig

        @render.plot(alt="Portfolio beta fan chart")
        def beta_fan_plot():
            try:
                df = mc_results()
            except Exception:
                df = None
            if df is None or df.empty:
                fig, ax = plt.subplots()
                ax.plot([])
                ax.set_title('Portfolio beta fan chart')
                ax.set_xlabel('Year')
                ax.set_ylabel('Beta')
                return fig
            q_df = compute_quantiles(df, 'beta_total')
            # Insert a baseline row for Year 0 corresponding to the starting portfolio beta.
            df_asset, _, _ = get_user_tables()
            nav0 = df_asset['Allocation'].sum()
            if nav0 > 0:
                beta0 = (df_asset['Allocation'] * df_asset['Beta'].astype(float)).sum() / nav0
            else:
                beta0 = 0.0
            if not q_df.empty:
                baseline_row = {'Year': 0}
                for col in q_df.columns[1:]:
                    baseline_row[col] = round(beta0, 2)
                q_df = pd.concat([pd.DataFrame([baseline_row]), q_df], ignore_index=True)
            fig, ax = plt.subplots()
            for col in q_df.columns[1:]:
                ax.plot(q_df['Year'], q_df[col], label=col)
            ax.set_title('Portfolio beta fan chart')
            ax.set_xlabel('Year')
            ax.set_ylabel('Beta')
            ax.legend()
            return fig

        @render.plot(alt="Portfolio private % fan chart")
        def private_fan_plot():
            try:
                df = mc_results()
            except Exception:
                df = None
            if df is None or df.empty:
                fig, ax = plt.subplots()
                ax.plot([])
                ax.set_title('Portfolio private % fan chart')
                ax.set_xlabel('Year')
                ax.set_ylabel('Private %')
                return fig
            q_df = compute_quantiles(df, 'private_total')
            # Insert a baseline row (Year 0) corresponding to the starting portfolio private fraction.
            df_asset, _, _ = get_user_tables()
            nav0 = df_asset['Allocation'].sum()
            # Local helper to convert percentage-like values to fractions
            def _to_fraction_local(val):
                try:
                    val_float = float(val)
                except Exception:
                    return 0.0
                if pd.isna(val_float):
                    return 0.0
                return val_float / 100.0 if abs(val_float) > 1.0 else val_float
            if nav0 > 0:
                private_frac_series = df_asset['Private %'].apply(_to_fraction_local)
                private0 = (df_asset['Allocation'] * private_frac_series).sum() / nav0
            else:
                private0 = 0.0
            if not q_df.empty:
                baseline_row = {'Year': 0}
                for col in q_df.columns[1:]:
                    baseline_row[col] = round(private0, 2)
                q_df = pd.concat([pd.DataFrame([baseline_row]), q_df], ignore_index=True)
            fig, ax = plt.subplots()
            for col in q_df.columns[1:]:
                ax.plot(q_df['Year'], q_df[col], label=col)
            ax.set_title('Portfolio private % fan chart')
            ax.set_xlabel('Year')
            ax.set_ylabel('Private %')
            ax.legend()
            return fig

    return server


# Instantiate the UI and server and bind to the app
app_ui = build_app_ui()
app_server = build_server()
app = App(app_ui, app_server)
