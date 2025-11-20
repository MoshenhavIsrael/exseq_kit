import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Dict, List, Tuple, Union

try:
    from scipy.stats import gaussian_kde
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


custom_palette = {
    'sick': 'red',
    'control': 'green'
}

custom_labels = {
    'sick': 'STXBP1',
    'control': 'Healthy'
}

def visualize_cell_features(cells_df_in, features=['num_punctas', 'cell_volume_voxels'], save_path=None):
    """
    Create histograms and boxplots for selected features, split by condition.
    """
    #
    cells_df = cells_df_in.copy()
    # Update labels for plotting
    cells_df['condition'] = cells_df['condition'].map(custom_labels).fillna(cells_df['condition'])
    cond_order = [custom_labels.get(k, k) for k in ['control', 'sick'] if custom_labels.get(k, k) in cells_df['condition'].unique()]
    custom_palette_new = {
        new_label: custom_palette[old_label]
        for old_label, new_label in custom_labels.items()
    }

    for feat in features:
        if feat not in cells_df.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histograms
        sns.histplot(
            data=cells_df, x=feat, hue='condition', kde=True, palette=custom_palette_new,
            hue_order=cond_order, stat='proportion', common_norm=False, ax=axes[1]
        )
        axes[1].set_title(f'Distribution of {feat} by condition')
        sns.histplot(
            data=cells_df, x=feat, hue='cell_type', kde=True,
            stat='proportion', common_norm=False, ax=axes[0]
        )
        axes[0].set_title(f'Distribution of {feat} by cell type')

        # Boxplot
        sns.violinplot(
            data=cells_df, x='cell_type', y=feat, hue='condition', palette=custom_palette_new,
            hue_order=cond_order, ax=axes[2]
        )
        axes[2].set_title(f'{feat} by Condition')

        plt.tight_layout()
        plt.show()

        if save_path is not None:
            fig.savefig(f"{save_path}/{feat}_distribution.png", dpi=300, bbox_inches="tight")


def plot_feature_distributions(
        df: pd.DataFrame,
        feature: str,
        hue: str,
        row_facet: Optional[str] = None,
        col_facet: Optional[str] = None,
        include_levels: Optional[Dict[str, List[str]]] = None,
        hue_order: Optional[List[str]] = None,
        row_order: Optional[List[str]] = None,
        col_order: Optional[List[str]] = None,
        palette: Optional[Union[Dict[str, str], List[str], str]] = None,
        stat: str = "proportion",
        common_norm: bool = False,
        bins: int = 40,
        clip_percentiles: Tuple[float, float] = (0, 100),
        xscale: str = "linear",  # {"linear", "log10"}
        epsilon: Optional[float] = None,  # if None -> auto epsilon
        epsilon_auto_quantile: float = 0.10,
        epsilon_auto_fraction: float = 0.01,
        dropna_mode: str = "drop",  # {"drop", "keep_as_missing"}
        legend_outside: bool = False,
        suptitle: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 5),
        grid_alpha: float = 0.15,
        kde: bool = True,
        line_width: float = 2.0,
        sharex: bool = True,
        sharey: bool = False,
        legend_count_mode: str = "global",
        xlabel_fontsize: Optional[float] = None,
        ylabel_fontsize: Optional[float] = None,
        save_path: Optional[str] = None,
):
    """
    Plot histograms (with consistent bins across panels) and log-space KDE (if requested)
    for a continuous feature, faceted by up to two categorical variables and colored by `hue`.

    Key design points:
    - Supports arbitrary categorical `hue` with >2 levels.
    - Supports faceting by `row_facet` and `col_facet`.
    - Handles NaN/Inf robustly and optionally maps NaN to a "Missing" category.
    - If xscale="log10": uses epsilon shift for stability and computes KDE in log-space with Jacobian
      so that the density curve is mathematically consistent with the histogram on the original X domain.
    - Adds (n=...) to legend labels for hue levels (global counts after filtering).
    - Ensures consistent binning across panels.

    Parameters
    ----------
    df : DataFrame
        Input table with at least `feature`, `hue` and optional facets.
    feature : str
        Name of the continuous feature column.
    hue : str
        Name of categorical column used as hue (can have many levels).
    row_facet, col_facet : Optional[str]
        Optional categorical columns for faceting into a grid.
    include_levels : Optional[Dict[str, List[str]]]
        Subset filter for specific categorical levels per column, e.g.,
        {"cell_type": ["immature", "mature"], "condition": ["control", "sick"]}.
        Levels not listed are excluded entirely before plotting.
    hue_order, row_order, col_order : Optional[List[str]]
        Explicit ordering of levels for hue and facets.
    palette : dict | list | str
        Color mapping for hue levels. If dict, keys must match hue levels. Otherwise, passed to seaborn.
    stat : {"density", "count", "proportion"}
        Histogram statistic. For rigorous alignment with KDE, "density" is recommended.
    common_norm : bool
        If True, normalize across hue levels jointly; otherwise each hue level normalized separately.
    bins : int
        Number of bins (consistent across panels). If xscale="log10", bins are uniform in log-space.
    clip_percentiles : tuple(float, float)
        Percentile-based clipping for global bin range computation (applied after filtering).
    xscale : {"linear", "log10"}
        X-axis scale. If "log10", epsilon shift and log-space KDE are applied.
    epsilon : Optional[float]
        Additive shift for log-scale stability. If None -> computed automatically from positive data.
    epsilon_auto_quantile : float
        Quantile (over positive values) used to scale epsilon when auto-computed.
    epsilon_auto_fraction : float
        Fraction of that quantile used for epsilon when auto-computed.
    dropna_mode : {"drop", "keep_as_missing"}
        How to handle NaN in categorical keys: drop rows vs. map to "Missing".
    legend_outside : bool
        Place a single figure-level legend outside the axes on the right.
    suptitle : Optional[str]
        Figure suptitle. If None, a default title is generated.
    figsize : (float, float)
        Figure size.
    grid_alpha : float
        Grid transparency per axes.
    kde : bool
        If True and SciPy is available, overlay KDE computed in log-space (when xscale="log10") or linear space otherwise.
    line_width : float
        Line width for KDE curves.
    sharex, sharey : bool
        Axes sharing flags across the facet grid.
    save_path : Optional[str]
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """

    import numpy as np

    # ---------------------------
    # 0) Basic validation
    # ---------------------------
    needed_cols = {feature, hue}
    if row_facet: needed_cols.add(row_facet)
    if col_facet: needed_cols.add(col_facet)
    missing = [c for c in needed_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    # Work on a copy to avoid mutating user's DataFrame
    data = df.copy()

    # ---------------------------
    # 1) Filter by include_levels (if provided)
    # ---------------------------
    if include_levels is not None:
        for col, levels in include_levels.items():
            if col not in data.columns:
                raise ValueError(f"'include_levels' refers to non-existent column: {col}")
            data = data[data[col].isin(levels)]

    # ---------------------------
    # 2) Handle NaN / Inf in feature and categorical keys
    # ---------------------------
    # Clean feature values
    data = data[np.isfinite(data[feature])]
    # Handle NaN in categorical keys
    cat_keys = [hue] + ([row_facet] if row_facet else []) + ([col_facet] if col_facet else [])
    if dropna_mode not in {"drop", "keep_as_missing"}:
        raise ValueError("dropna_mode must be one of {'drop', 'keep_as_missing'}")

    if dropna_mode == "drop":
        for c in cat_keys:
            data = data[data[c].notna()]
    else:
        for c in cat_keys:
            data.loc[:, c] = data[c].astype(object)
            data.loc[data[c].isna(), c] = "Missing"

    # If no data remains, return early with an informative error
    if len(data) == 0:
        raise ValueError("No data to plot after filtering/cleaning.")

    # ---------------------------
    # 3) Establish categorical orders and palette
    # ---------------------------
    def _levels_in_order(series: pd.Series, explicit_order: Optional[List[str]]):
        if explicit_order is not None:
            levels = [lvl for lvl in explicit_order if lvl in series.unique().tolist()]
        else:
            # Stable order by frequency (desc) then alphabetically as tiebreaker
            counts = series.value_counts(dropna=False)
            levels = counts.index.tolist()
        return levels

    if hue_order is None:
        hue_order = _levels_in_order(data[hue], None)

    if row_facet:
        if row_order is None:
            row_order = _levels_in_order(data[row_facet], None)
    if col_facet:
        if col_order is None:
            col_order = _levels_in_order(data[col_facet], None)

    # Ensure categorical dtype with the chosen order (for consistent legends/facets)
    data[hue] = pd.Categorical(data[hue], categories=hue_order, ordered=True)
    if row_facet:
        data[row_facet] = pd.Categorical(data[row_facet], categories=row_order, ordered=True)
    if col_facet:
        data[col_facet] = pd.Categorical(data[col_facet], categories=col_order, ordered=True)

    # Build a hue->color mapping if a dict was not supplied
    if isinstance(palette, dict):
        hue_colors = {lvl: palette.get(lvl, None) for lvl in hue_order}
    else:
        # Let seaborn handle cycling, but we still build a mapping to use for KDE lines
        color_cycle = sns.color_palette(palette, n_colors=len(hue_order)) if palette is not None \
            else sns.color_palette(n_colors=len(hue_order))
        hue_colors = {lvl: color_cycle[i] for i, lvl in enumerate(hue_order)}

    # ---------------------------
    # 4) Compute global histogram bin edges (consistent across panels)
    # ---------------------------
    # Percentile clipping for robust range
    lo, hi = np.nanpercentile(data[feature].values, clip_percentiles)
    # Guard against degenerate ranges
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(data[feature]))
        hi = float(np.nanmax(data[feature]))
    if lo == hi:
        lo, hi = lo * 0.99, hi * 1.01  # force a small non-zero span

    if xscale == "log10":
        # Auto epsilon if not provided
        if epsilon is None:
            positives = data.loc[data[feature] > 0, feature].values
            if len(positives) == 0:
                raise ValueError("All feature values are non-positive; cannot use log10 scale.")
            base_q = np.quantile(positives, epsilon_auto_quantile)
            auto_eps = max(1e-12, epsilon_auto_fraction * base_q)
            epsilon = float(auto_eps)
        # Effective domain for bins in original x with epsilon shift
        lo_eff = max(lo + epsilon, 1e-12)
        hi_eff = max(hi + epsilon, lo_eff * 1.01)
        # Log-space uniform bins, then map back to original scale (subtract epsilon for labeling)
        log_edges = np.linspace(np.log10(lo_eff), np.log10(hi_eff), bins + 1)
        bin_edges = (10.0 ** log_edges) - epsilon
        # Prevent negative labels due to epsilon subtraction (clip to zero)
        bin_edges = np.clip(bin_edges, a_min=0.0, a_max=None)
    else:
        bin_edges = np.linspace(lo, hi, bins + 1)

    # ---------------------------
    # 5) Build the facet grid axes
    # ---------------------------
    n_rows = len(row_order) if row_facet else 1
    n_cols = len(col_order) if col_facet else 1

    # Determine effective sharex status: MUST be False for log10 scale to avoid rendering bugs
    # due to inconsistent scaling inheritance between axes.
    effective_sharex = sharex and (xscale != "log10")

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=figsize,
                             sharex=effective_sharex,  # Use the calculated effective status
                             sharey=sharey)
    if isinstance(axes, np.ndarray):
        axes_arr = axes
    else:
        axes_arr = np.array([[axes]])
    axes_flat = axes_arr.ravel()

    # ---------------------------
    # 6) Precompute global hue counts for legend labels
    # ---------------------------
    if legend_count_mode == "global":
        global_counts = data.groupby(hue).size().reindex(hue_order, fill_value=0).to_dict()
        legend_labels = [f"{lvl} (n={global_counts.get(lvl, 0)})" for lvl in hue_order]
    else:
        # Define empty global labels, they will be computed per panel
        global_counts = None
        legend_labels = None  # Will be computed later or not used    # 7) Helper: draw a single panel (hist + optional KDE)

    # ---------------------------
    # 7) Helper: draw a single panel (hist + optional KDE)
    # ---------------------------
    def _draw_panel(ax, subset: pd.DataFrame, panel_title: str):

        # Histogram(s)
        # Determine if we should use Seaborn's internal KDE (for proportion/count)
        # or if we need to suppress it to run our manual density/Jacobian KDE.
        use_internal_kde = kde and (stat != "density")

        sns.histplot(
            data=subset,
            x=feature,
            hue=hue,
            hue_order=hue_order,
            bins=bin_edges,
            stat=stat,
            common_norm=common_norm,
            palette=hue_colors,
            kde=use_internal_kde,  # Use internal KDE only if stat is not 'density'
            legend=True,  # Ensure artists are created for the legend
            ax=ax
        )

        # MANUAL/JACOBIAN KDE overlay (Only run if stat='density' and kde is true)
        if kde and stat == "density" and _HAVE_SCIPY:

            # Use the mathematically correct (Jacobian) KDE method

            # Build an evaluation grid on original x-domain (respecting xscale)
            if xscale == "log10":
                x_min = max(np.min(bin_edges), 0.0)
                x_max = np.max(bin_edges)
                # Ensure positive after epsilon shift
                x_eval_eff = np.logspace(
                    np.log10(max(x_min + epsilon, 1e-12)),
                    np.log10(x_max + epsilon),
                    num=400
                )  # in shifted domain
                x_eval = x_eval_eff - epsilon
                x_eval = np.clip(x_eval, a_min=0.0, a_max=None)
            else:
                x_eval = np.linspace(np.min(bin_edges), np.max(bin_edges), 400)

            # For each hue level, compute KDE and plot
            for lvl in hue_order:
                x_lvl = subset.loc[subset[hue] == lvl, feature].values
                if len(x_lvl) < 2:
                    continue  # too few points for KDE

                if xscale == "log10":
                    # KDE in log-space with epsilon shift: y = log10(x + eps)
                    y_vals = np.log10(x_lvl + epsilon)
                    # Guard against non-positive after shift (exclude very negative cases)
                    y_vals = y_vals[np.isfinite(y_vals)]
                    if len(y_vals) < 2:
                        continue
                    kde_y = gaussian_kde(y_vals)
                    # Evaluate in log-space at y(x) and transform back via Jacobian:
                    y_eval = np.log10(x_eval + epsilon)
                    f_y = kde_y(y_eval)  # density in y-space
                    # f_x(x) = f_y(y) * dy/dx ; with y = log10(x+eps) => dy/dx = 1 / ((x+eps)*ln(10))
                    jac = 1.0 / ((x_eval + epsilon) * np.log(10.0))
                    f_x = f_y * jac
                else:
                    # Linear-scale KDE directly on x
                    x_vals = x_lvl[np.isfinite(x_lvl)]
                    if len(x_vals) < 2:
                        continue
                    kde_x = gaussian_kde(x_vals)
                    f_x = kde_x(x_eval)

                # If histogram stat is not "density", optionally rescale for visual comparability
                # Strictly speaking, KDE is a density; for "proportion" or "count" we draw the curve as-is
                # to preserve its shape, without pretending it's the same unit.
                ax.plot(x_eval, f_x, linewidth=line_width, color=hue_colors.get(lvl, None), label=None)

        # Titles and cosmetics
        ax.set_title(panel_title)

        # Apply custom font size if specified
        ax.set_xlabel(feature,
                      fontsize=xlabel_fontsize if xlabel_fontsize is not None else plt.rcParams['axes.labelsize'])
        ylabel = {"density": "Density", "count": "Count", "proportion": "Proportion"}.get(stat, "Value")
        # Apply custom font size if specified
        ax.set_ylabel(ylabel,
                      fontsize=ylabel_fontsize if ylabel_fontsize is not None else plt.rcParams['axes.labelsize'])
        ax.grid(True, alpha=grid_alpha)

        # Axis scale
        if xscale == "log10":
            ax.set_xscale("log")

    # ---------------------------
    # 8) Iterate panels and draw
    # ---------------------------
    # (Handles are now custom-generated, so we only need to track the palette/order)
    from matplotlib.patches import Rectangle

    import numpy as np
    axes_arr = np.ravel(axes_arr)

    for i_r in range(n_rows):
        for i_c in range(n_cols):

            linear_index = i_r * n_cols + i_c
            ax = axes_arr[linear_index]

            # 8.1) Filter data based on facets (existing logic)
            if row_facet and col_facet:
                r_val = row_order[i_r]
                c_val = col_order[i_c]
                panel_df = data[(data[row_facet] == r_val) & (data[col_facet] == c_val)]
                panel_title = f"{row_facet}={r_val} | {col_facet}={c_val}"
            elif row_facet:
                r_val = row_order[i_r]
                if i_c > 0:
                    ax.set_visible(False)
                    continue
                panel_df = data[data[row_facet] == r_val]
                panel_title = f"{row_facet}={r_val}"
            elif col_facet:
                c_val = col_order[i_c]
                if i_r > 0:
                    ax.set_visible(False)
                    continue
                panel_df = data[data[col_facet] == c_val]
                panel_title = f"{col_facet}={c_val}"
            else:
                if i_r > 0 or i_c > 0:
                    ax.set_visible(False)
                    continue
                panel_df = data
                panel_title = ""

            # 8.2) Handle empty panel
            if len(panel_df) == 0:
                ax.set_visible(True)
                ax.set_title(panel_title + " (no data)")
                ax.axis("off")
                continue

            # 8.3) Draw panel
            _draw_panel(ax, panel_df, panel_title)

            # 8.4) LEGEND LOGIC: Per-panel counts and legend
            if legend_count_mode == "panel":

                # Compute counts ONLY for the current panel
                panel_counts = panel_df.groupby(hue).size().reindex(hue_order, fill_value=0).to_dict()
                panel_labels = [f"{lvl} (n={panel_counts.get(lvl, 0)})" for lvl in hue_order]

                # Generate custom handles (colored squares)
                custom_handles = [
                    Rectangle((0, 0), 1, 1, fc=hue_colors.get(lvl, 'gray'))
                    for lvl in hue_order
                ]

                # Add legend to the current axes
                ax.legend(
                    handles=custom_handles,
                    labels=panel_labels,
                    loc="best",
                    frameon=True,
                    title=hue,
                    # Ensure no duplicate titles are shown if KDE was drawn with labels
                    # Although we removed the explicit label, this is a safety measure
                    # for consistency with the intended behavior of a panel-level legend.
                    # This call WILL override any legend created by seaborn/matplotlib internally.
                )

            # Remove any pre-existing legend if the mode is GLOBAL (or no facet)
            elif legend_count_mode == "global":
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

    # ---------------------------
    # 9) Single legend with (n=...) labels (Only for GLOBAL mode)
    # ---------------------------

    if legend_count_mode == "global" and len(data) > 0:
        from matplotlib.patches import Rectangle

        # 1. Generate Custom Handles (colored squares) - uses global hue_order
        custom_handles = [
            Rectangle((0, 0), 1, 1, fc=hue_colors.get(lvl, 'gray'))
            for lvl in hue_order
        ]

        # 2. Get the Labels (computed in step 6 with global n=...)
        # We reuse the existing 'legend_labels' list

        # 3. Draw the legend
        if legend_outside:
            fig.legend(
                handles=custom_handles,
                labels=legend_labels,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                frameon=True,
                title=hue
            )
        else:
            # Place on the first axes by default
            axes_arr.flat[0].legend(
                handles=custom_handles,
                labels=legend_labels,
                loc="best",
                frameon=True,
                title=hue
            )

    # ---------------------------
    # 10) Suptitle and footnote (epsilon/log info)
    # ---------------------------
    if suptitle is None:
        suptitle = f"{feature} distributions by {hue}"
        if row_facet or col_facet:
            facets = " × ".join([v for v in [row_facet, col_facet] if v is not None])
            suptitle += f" | facets: {facets}"
    fig.suptitle(suptitle, y=0.98, fontsize=14)

    # # Optional footnote with epsilon/clip info for log-scale
    # if xscale == "log10":
    #     foot = f"log10-scale with epsilon={epsilon:.3e}; clip={clip_percentiles[0]}–{clip_percentiles[1]} percentiles"
    #     fig.text(0.01, 0.01, foot, fontsize=9, alpha=0.75)

    fig.tight_layout(rect=[0, 0, 0.98, 0.965] if legend_outside else [0, 0, 1, 0.965])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_correlation_scatter(
        df: pd.DataFrame,
        feature_x: str,
        feature_y: str,
        hue: Optional[str] = None,
        hue_order: Optional[List[str]] = None,
        palette: Optional[Union[Dict[str, str], List[str], str]] = None,
        log_x: bool = False,
        log_y: bool = False,
        suptitle: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 6),
        xlabel_fontsize: Optional[float] = None,
        ylabel_fontsize: Optional[float] = None,
        save_path: Optional[str] = None,
):
    """
    Plots a scatter plot for two features with a linear regression line
    (trend line) and displays the overall Pearson correlation coefficient,
    as well as per-group correlations if 'hue' is specified.

    Parameters
    ----------
    df : pd.DataFrame
        Input data table.
    feature_x : str
        Column name for the X-axis.
    feature_y : str
        Column name for the Y-axis.
    hue : Optional[str]
        Column name for grouping and coloring the data points.
    hue_order : Optional[List[str]]
        Explicit order for the hue levels.
    palette : Optional[Union[Dict[str, str], List[str], str]]
        Color mapping for hue levels.
    log_x : bool
        If True, set X-axis scale to logarithmic.
    log_y : bool
        If True, set Y-axis scale to logarithmic.
    suptitle : Optional[str]
        Figure super title.
    figsize : Tuple[float, float]
        Figure size.
    save_path : Optional[str]
        Path to save the figure.
    """

    # 1. Data Preparation and Cleaning
    cols_to_check = [feature_x, feature_y]
    if hue:
        cols_to_check.append(hue)

    plot_data = df.copy().dropna(subset=cols_to_check)

    if plot_data.empty:
        raise ValueError("No valid data points remain after dropping NaNs.")

    # 2. Overall Correlation Calculation (Global)
    # Calculate Pearson's correlation coefficient for the entire dataset
    correlation_global = plot_data[[feature_x, feature_y]].corr().iloc[0, 1]

    # 3. Determine Hue Order and Color Map
    if hue and hue_order is None:
        hue_order = plot_data[hue].value_counts().index.tolist()

    if hue and isinstance(palette, dict):
        hue_colors = palette
    elif hue:
        n_colors = len(hue_order)
        color_cycle = sns.color_palette(palette, n_colors=n_colors) if palette is not None \
            else sns.color_palette(n_colors=n_colors)
        hue_colors = {lvl: color_cycle[i] for i, lvl in enumerate(hue_order)}
    else:
        hue_colors = {None: 'darkblue'}

    # 4. Plotting Setup
    fig, ax = plt.subplots(figsize=figsize)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # 5. Draw Scatter Points
    sns.scatterplot(
        data=plot_data,
        x=feature_x,
        y=feature_y,
        hue=hue,
        hue_order=hue_order,
        palette=hue_colors if hue else {None: hue_colors[None]},
        ax=ax,
        alpha=0.6,
        legend='full' if hue else False
    )

    # 6. Draw Trend Line(s) and Calculate Per-Group Correlation
    group_correlations = {}  # Dictionary to store per-group results

    if hue:
        # If hue is used, groups is a pandas GroupBy object
        groups = plot_data.groupby(hue)
        trend_groups = hue_order
    else:
        # If no hue, groups is a dictionary holding the entire data
        groups = {None: plot_data}
        trend_groups = [None]  # Iterate only once for the whole data

    for lvl in trend_groups:

        # Access the subset data based on whether 'hue' is used
        if hue:
            # Use get_group() for pandas GroupBy object
            try:
                subset = groups.get_group(lvl)
            except KeyError:
                # Handle cases where a level in hue_order might be empty in plot_data
                continue
        else:
            # Use dictionary access for the single global group
            subset = groups.get(lvl, pd.DataFrame())

        if subset.empty:
            continue

        # Calculate per-group correlation
        correlation_group = subset[[feature_x, feature_y]].corr().iloc[0, 1]
        group_correlations[lvl] = correlation_group

        # Determine color for the trend line
        line_color = hue_colors.get(lvl, 'gray') if hue else 'red'

        # Draw a linear regression line for the subset
        sns.regplot(
            data=subset,
            x=feature_x,
            y=feature_y,
            ax=ax,
            scatter=False,
            color=line_color,
            line_kws={'linestyle': '-', 'linewidth': 2},
            logx=log_x
        )

    # 7. Final Touches

    # 7.1 Construct correlation text box (Global + Per-Group)
    corr_lines = [f"Overall $\\rho$: {correlation_global:.3f}"]

    if hue:
        corr_lines.append("\nPer Group $\\rho$:")
        # Append group correlations in order
        for lvl in hue_order:
            if lvl in group_correlations:
                corr_lines.append(f"  {lvl}: {group_correlations[lvl]:.3f}")

    final_corr_text = "\n".join(corr_lines)

    # 7.2 Add correlation text to the plot
    ax.text(
        0.95, 0.95,
        final_corr_text,
        transform=ax.transAxes,
        fontsize=10 if hue else 12,  # Adjust font size if multiple lines are present
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.85)
    )

    # Apply custom font size if specified
    ax.set_xlabel(feature_x,
                  fontsize=xlabel_fontsize if xlabel_fontsize is not None else plt.rcParams['axes.labelsize'])
    # Apply custom font size if specified
    ax.set_ylabel(feature_y,
                  fontsize=ylabel_fontsize if ylabel_fontsize is not None else plt.rcParams['axes.labelsize'])

    ax.set_title(f"Scatter Plot: {feature_y} vs. {feature_x}")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    plt.show()

from typing import Optional, Union, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def binned_summary_plot(
    df: pd.DataFrame,
    x: str,                      # X column (e.g., 'compactness')
    y: str,                      # Y column (e.g., 'puncta_count')
    group: Optional[str] = None, # Optional grouping column (e.g., 'condition')
    facet: Optional[str] = None, # Optional faceting column (small multiples)
    bins: Union[str, Sequence[float]] = "quantile",  # 'quantile' | 'uniform' | explicit bin edges
    n_bins: int = 10,            # Number of bins for 'quantile'/'uniform'
    share_bins: bool = True,     # Use identical bin edges across groups (per facet if facet!=None)
    min_per_bin: int = 5,        # Minimum N per (group, bin) to report
    agg: str = "mean",           # 'mean' | 'median' (line)
    spread: str = "sd",          # 'sd' | 'sem' | 'iqr' (error band)
    normalize_y_by: Optional[str] = None,  # Column to divide Y by (e.g., area/volume)
    transform_y: Optional[Union[str, callable]] = None,  # None | 'log1p' | callable f(y)
    show_counts: bool = True,    # Show background bars with per-bin N on secondary axis
    suptitle: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    xlabel_fontsize: Optional[float] = None,
    ylabel_fontsize: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    colors: Optional[dict] = None,  # Mapping: group value -> color (consistent across facets)
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray], pd.DataFrame]:
    """
    Bin X, summarize Y within each bin (optionally per group), and plot Y_agg (mean/median)
    with an error band (sd/sem/iqr). Supports faceting by a categorical column.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or np.ndarray of Axes (if faceted)
    summary_df : pd.DataFrame
        Columns include (when present):
          ['facet', 'group', 'bin_idx', 'x_left', 'x_right', 'x_center',
           'n', 'y_mean', 'y_variance', 'y_agg', 'y_spread_low', 'y_spread_high']
    """
    # ---------- input checks ----------
    needed = [x, y]
    if group: needed.append(group)
    if facet: needed.append(facet)
    if normalize_y_by: needed.append(normalize_y_by)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be one of {'mean','median'}.")
    if spread not in {"sd", "sem", "iqr"}:
        raise ValueError("spread must be one of {'sd','sem','iqr'}.")

    # ---------- base prep ----------
    cols = [x, y]
    if group: cols.append(group)
    if facet: cols.append(facet)
    if normalize_y_by: cols.append(normalize_y_by)
    work_all = df[cols].copy().dropna(subset=[x, y])

    # normalize Y if requested
    def _normalize_and_transform(_df: pd.DataFrame) -> np.ndarray:
        y_vals = _df[y].astype(float).to_numpy()
        if normalize_y_by:
            denom = _df[normalize_y_by].astype(float).replace(0, np.nan).to_numpy()
            y_vals = y_vals / denom
        if transform_y is not None:
            if transform_y == "log1p":
                y_vals = np.log1p(y_vals)
            elif callable(transform_y):
                y_vals = transform_y(y_vals)
            else:
                raise ValueError("transform_y must be None, 'log1p', or a callable.")
        return y_vals

    # bin edges computation
    def _compute_edges(values: np.ndarray) -> np.ndarray:
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.array([0, 1], dtype=float)
        if isinstance(bins, str):
            if bins == "quantile":
                qs = np.linspace(0, 1, n_bins + 1)
                edges = np.quantile(values, qs)
                edges[0] = np.min(values)
                edges[-1] = np.max(values)
            elif bins == "uniform":
                vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
                if vmin == vmax:
                    vmin -= 0.5
                    vmax += 0.5
                edges = np.linspace(vmin, vmax, n_bins + 1)
            else:
                raise ValueError("bins must be 'quantile', 'uniform', or explicit edges.")
        else:
            edges = np.asarray(bins, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("Explicit bins must be a 1D sequence with length >= 2.")
        edges = np.unique(edges)
        if edges.size < 2:
            e0 = values.min()
            e1 = values.max() + 1e-9
            edges = np.array([e0, e1], dtype=float)
        return edges

    def _assign_bins(series: pd.Series, edges_arr: np.ndarray) -> pd.Categorical:
        return pd.cut(series, bins=edges_arr, include_lowest=True)

    # summarization per (group, bin) for a given dataframe slice
    def _summarize_slice(sdf: pd.DataFrame, facet_value=None) -> pd.DataFrame:
        sdf = sdf.copy()
        sdf["_Y_"] = _normalize_and_transform(sdf)

        # edges per slice (and shared across groups within this slice if share_bins)
        if (group is None) or share_bins:
            edges_local = _compute_edges(sdf[x].to_numpy())
            edge_map_local = None
        else:
            edge_map_local = {gval: _compute_edges(gdf[x].to_numpy())
                              for gval, gdf in sdf.groupby(group)}
            edges_local = None

        rows = []
        if group is None:
            bins_cat = _assign_bins(sdf[x], edges_local)
            for i, cat in enumerate(bins_cat.cat.categories):
                mask = (bins_cat == cat)
                n = int(mask.sum())
                x_left, x_right = float(cat.left), float(cat.right)
                x_center = 0.5 * (x_left + x_right)
                if n < min_per_bin:
                    rows.append({
                        "facet": facet_value, "group": None, "bin_idx": i,
                        "x_left": x_left, "x_right": x_right, "x_center": x_center,
                        "n": n, "y_mean": np.nan, "y_variance": np.nan,
                        "y_agg": np.nan, "y_spread_low": np.nan, "y_spread_high": np.nan
                    })
                    continue
                ybin = sdf.loc[mask, "_Y_"].to_numpy()
                y_mean = float(np.nanmean(ybin))
                y_var = float(np.nanvar(ybin, ddof=1)) if n > 1 else 0.0
                y_agg = y_mean if agg == "mean" else float(np.nanmedian(ybin))
                if spread == "sd":
                    s = float(np.nanstd(ybin, ddof=1)) if n > 1 else 0.0
                    low, high = y_agg - s, y_agg + s
                elif spread == "sem":
                    sd = float(np.nanstd(ybin, ddof=1)) if n > 1 else 0.0
                    s = sd / np.sqrt(n) if n > 0 else np.nan
                    low, high = y_agg - s, y_agg + s
                else:
                    q1, q3 = np.nanpercentile(ybin, [25, 75])
                    low, high = float(q1), float(q3)
                rows.append({
                    "facet": facet_value, "group": None, "bin_idx": i,
                    "x_left": x_left, "x_right": x_right, "x_center": x_center,
                    "n": n, "y_mean": y_mean, "y_variance": y_var,
                    "y_agg": float(y_agg), "y_spread_low": float(low), "y_spread_high": float(high)
                })
            return pd.DataFrame(rows), edges_local

        else:
            all_rows = []
            if share_bins:
                g_edges = edges_local
                for gval, gdf in sdf.groupby(group):
                    bins_cat = _assign_bins(gdf[x], g_edges)
                    for i, cat in enumerate(bins_cat.cat.categories):
                        mask = (bins_cat == cat)
                        n = int(mask.sum())
                        x_left, x_right = float(cat.left), float(cat.right)
                        x_center = 0.5 * (x_left + x_right)
                        if n < min_per_bin:
                            all_rows.append({
                                "facet": facet_value, "group": gval, "bin_idx": i,
                                "x_left": x_left, "x_right": x_right, "x_center": x_center,
                                "n": n, "y_mean": np.nan, "y_variance": np.nan,
                                "y_agg": np.nan, "y_spread_low": np.nan, "y_spread_high": np.nan
                            })
                            continue
                        ybin = gdf.loc[mask, "_Y_"].to_numpy()
                        y_mean = float(np.nanmean(ybin))
                        y_var = float(np.nanvar(ybin, ddof=1)) if n > 1 else 0.0
                        y_agg = y_mean if agg == "mean" else float(np.nanmedian(ybin))
                        if spread == "sd":
                            s = float(np.nanstd(ybin, ddof=1)) if n > 1 else 0.0
                            low, high = y_agg - s, y_agg + s
                        elif spread == "sem":
                            sd = float(np.nanstd(ybin, ddof=1)) if n > 1 else 0.0
                            s = sd / np.sqrt(n) if n > 0 else np.nan
                            low, high = y_agg - s, y_agg + s
                        else:
                            q1, q3 = np.nanpercentile(ybin, [25, 75])
                            low, high = float(q1), float(q3)
                        all_rows.append({
                            "facet": facet_value, "group": gval, "bin_idx": i,
                            "x_left": x_left, "x_right": x_right, "x_center": x_center,
                            "n": n, "y_mean": y_mean, "y_variance": y_var,
                            "y_agg": float(y_agg), "y_spread_low": float(low), "y_spread_high": float(high)
                        })
                return pd.DataFrame(all_rows), g_edges
            else:
                for gval, gdf in sdf.groupby(group):
                    g_edges = edge_map_local[gval]
                    bins_cat = _assign_bins(gdf[x], g_edges)
                    for i, cat in enumerate(bins_cat.cat.categories):
                        mask = (bins_cat == cat)
                        n = int(mask.sum())
                        x_left, x_right = float(cat.left), float(cat.right)
                        x_center = 0.5 * (x_left + x_right)
                        if n < min_per_bin:
                            all_rows.append({
                                "facet": facet_value, "group": gval, "bin_idx": i,
                                "x_left": x_left, "x_right": x_right, "x_center": x_center,
                                "n": n, "y_mean": np.nan, "y_variance": np.nan,
                                "y_agg": np.nan, "y_spread_low": np.nan, "y_spread_high": np.nan
                            })
                            continue
                        ybin = gdf.loc[mask, "_Y_"].to_numpy()
                        y_mean = float(np.nanmean(ybin))
                        y_var = float(np.nanvar(ybin, ddof=1)) if n > 1 else 0.0
                        y_agg = y_mean if agg == "mean" else float(np.nanmedian(ybin))
                        if spread == "sd":
                            s = float(np.nanstd(ybin, ddof=1)) if n > 1 else 0.0
                            low, high = y_agg - s, y_agg + s
                        elif spread == "sem":
                            sd = float(np.nanstd(ybin, ddof=1)) if n > 1 else 0.0
                            s = sd / np.sqrt(n) if n > 0 else np.nan
                            low, high = y_agg - s, y_agg + s
                        else:
                            q1, q3 = np.nanpercentile(ybin, [25, 75])
                            low, high = float(q1), float(q3)
                        all_rows.append({
                            "facet": facet_value, "group": gval, "bin_idx": i,
                            "x_left": x_left, "x_right": x_right, "x_center": x_center,
                            "n": n, "y_mean": y_mean, "y_variance": y_var,
                            "y_agg": float(y_agg), "y_spread_low": float(low), "y_spread_high": float(high)
                        })
                return pd.DataFrame(all_rows), None

    # ---------- faceting logic ----------
    if facet is None:
        # single axes
        summary_df, edges_used = _summarize_slice(work_all, facet_value=None)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        # counts background
        if show_counts:
            counts_edges = edges_used if edges_used is not None else _compute_edges(work_all[x].to_numpy())
            counts_bins = pd.cut(work_all[x], bins=counts_edges, include_lowest=True)
            counts = counts_bins.value_counts(sort=False).values
            centers = 0.5 * (counts_edges[:-1] + counts_edges[1:])
            if counts.size > 0:
                ax2 = ax.twinx()
                width = (counts_edges[1] - counts_edges[0]) * 0.85 if len(counts_edges) > 1 else 0.1
                ax2.bar(centers, counts, width=width, alpha=0.15, edgecolor='none')
                ax2.set_ylabel("N per bin", fontsize=(ylabel_fontsize or 10))
                ax2.grid(False)

        # lines + ribbons (match color to line with alpha)
        if group is None:
            sdf = summary_df.sort_values("bin_idx")
            line, = ax.plot(sdf["x_center"], sdf["y_agg"], label=y) # , marker="o"
            c = line.get_color()
            ax.fill_between(sdf["x_center"], sdf["y_spread_low"], sdf["y_spread_high"], alpha=0.2, color=c)
        else:
            groups_unique = summary_df["group"].dropna().astype(str).unique()
            # consistent colors
            color_map = {}
            if colors:
                color_map.update(colors)
            else:
                # use matplotlib default cycle
                cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
                for i, gval in enumerate(groups_unique):
                    color_map[gval] = cycle[i % len(cycle)] if cycle else None

            for gval in groups_unique:
                sdf = summary_df.loc[summary_df["group"] == gval].sort_values("bin_idx")
                c = color_map.get(gval, None)
                ax.plot(sdf["x_center"], sdf["y_agg"], label=str(gval), color=c) # , marker="o"
                ax.fill_between(sdf["x_center"], sdf["y_spread_low"], sdf["y_spread_high"], alpha=0.2, color=c)

        ax.set_xlabel(x, fontsize=xlabel_fontsize)
        y_label_core = y if not normalize_y_by else f"{y} / {normalize_y_by}"
        if transform_y == "log1p":
            y_label_core = f"log1p({y_label_core})"
        ax.set_ylabel(f"{'Mean' if agg=='mean' else 'Median'} ± {spread.upper()}", fontsize=ylabel_fontsize)
        ax.grid(True, alpha=0.3)
        if group is not None:
            ax.legend(frameon=False, title=(group or None))

        if suptitle:
            fig.suptitle(suptitle)

        # tidy summary_df
        order_cols = ["facet", "group", "bin_idx", "x_left", "x_right", "x_center", "n",
                      "y_mean", "y_variance", "y_agg", "y_spread_low", "y_spread_high"]
        summary_df = summary_df[[c for c in order_cols if c in summary_df.columns]].sort_values(
            ["bin_idx"] if "group" not in summary_df.columns else ["group", "bin_idx"]
        ).reset_index(drop=True)

        return fig, ax, summary_df

    # faceted case
    facet_vals = work_all[facet].astype(str).unique()
    n_facets = len(facet_vals)
    ncols = int(np.ceil(np.sqrt(n_facets)))
    nrows = int(np.ceil(n_facets / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    all_summaries = []

    # consistent group colors across facets
    global_groups = []
    if group:
        global_groups = work_all[group].astype(str).unique()
    color_map = {}
    if colors:
        color_map.update(colors)
    else:
        cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        for i, gval in enumerate(global_groups):
            color_map[gval] = cycle[i % len(cycle)] if cycle else None

    for idx, fval in enumerate(sorted(facet_vals)):
        ax_i = axes_flat[idx]
        sub = work_all.loc[work_all[facet].astype(str) == fval]
        summary_slice, edges_used = _summarize_slice(sub, facet_value=fval)
        all_summaries.append(summary_slice)

        # counts background per facet
        if show_counts:
            counts_edges = edges_used if edges_used is not None else _compute_edges(sub[x].to_numpy())
            counts_bins = pd.cut(sub[x], bins=counts_edges, include_lowest=True)
            counts = counts_bins.value_counts(sort=False).values
            centers = 0.5 * (counts_edges[:-1] + counts_edges[1:])
            if counts.size > 0:
                ax2 = ax_i.twinx()
                width = (counts_edges[1] - counts_edges[0]) * 0.85 if len(counts_edges) > 1 else 0.1
                ax2.bar(centers, counts, width=width, alpha=0.15, edgecolor='none')
                ax2.set_ylabel("N per bin", fontsize=(ylabel_fontsize or 10))
                ax2.grid(False)

        # plot lines+ribbons on this facet
        if group is None:
            sdf = summary_slice.sort_values("bin_idx")
            line, = ax_i.plot(sdf["x_center"], sdf["y_agg"]) # , marker="o"
            c = line.get_color()
            ax_i.fill_between(sdf["x_center"], sdf["y_spread_low"], sdf["y_spread_high"], alpha=0.2, color=c)
        else:
            for gval in summary_slice["group"].dropna().astype(str).unique():
                sdf = summary_slice.loc[summary_slice["group"] == gval].sort_values("bin_idx")
                c = color_map.get(gval, None)
                ax_i.plot(sdf["x_center"], sdf["y_agg"], label=str(gval), color=c) # , marker="o"
                ax_i.fill_between(sdf["x_center"], sdf["y_spread_low"], sdf["y_spread_high"], alpha=0.2, color=c)

        ax_i.set_title(f"{facet} = {fval}")
        ax_i.set_xlabel(x, fontsize=xlabel_fontsize)
        y_label_core = y if not normalize_y_by else f"{y} / {normalize_y_by}"
        if transform_y == "log1p":
            y_label_core = f"log1p({y_label_core})"
        ax_i.set_ylabel(f"{'Mean' if agg=='mean' else 'Median'} ± {spread.upper()}", fontsize=ylabel_fontsize)
        ax_i.grid(True, alpha=0.3)
        if group is not None:
            ax_i.legend(frameon=False, title=(group or None))

    # hide any unused subplot axes
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    order_cols = ["facet", "group", "bin_idx", "x_left", "x_right", "x_center", "n",
                  "y_mean", "y_variance", "y_agg", "y_spread_low", "y_spread_high"]
    summary_df = summary_df[[c for c in order_cols if c in summary_df.columns]].sort_values(
        ["facet", "bin_idx"] if "group" not in summary_df.columns else ["facet", "group", "bin_idx"]
    ).reset_index(drop=True)

    return fig, axes, summary_df
