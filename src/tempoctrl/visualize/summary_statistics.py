"""Visualizations for summary statistics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from matplotlib.ticker import StrMethodFormatter


def _to_pandas(dataframe: Any) -> pd.DataFrame:
    """Return a pandas representation of a dataframe-like object."""
    if isinstance(dataframe, pd.DataFrame):
        return dataframe

    to_pandas = getattr(dataframe, "to_pandas", None)
    if callable(to_pandas):
        converted = to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted

    try:
        return pd.DataFrame(dataframe)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "dataframe must be a pandas dataframe or an object that can be "
            "converted to one"
        ) from exc


def plot_histogram(
    dataframe: Any,
    continuous_variable: str,
    *,
    title: str | None = None,
    title_fontsize: str = "large",
    x_label: str | None = None,
    y_label: str | None = None,
    histogram_color: str = "skyblue",
    histogram_edgecolor: str = "white",
    bins: int = 30,
    alpha: float = 0.7,
    figsize: tuple[float, float] = (10.0, 6.0),
    grid: bool = True,
    grid_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """Plot a histogram for a continuous variable in a dataframe.

    The function accepts pandas dataframes, Polars dataframes, and other
    dataframe-like objects that pandas can convert. Missing observations are
    ignored by seaborn.

    Args:
        dataframe: Data containing the variable to plot.
        continuous_variable: Name of the numeric column to plot.
        title: Plot title. Defaults to ``"Distribution of <column>"``.
        title_fontsize: Matplotlib font-size name for the title. The
            spelling is retained for compatibility with the requested
            public API.
        x_label: X-axis label. Defaults to the continuous variable's name.
        y_label: Y-axis label. Defaults to ``"Count"``.
        histogram_color: Matplotlib-compatible bar color.
        histogram_edgecolor: Matplotlib-compatible bar edge color.
        bins: Number of equal-width histogram bins.
        alpha: Bar opacity between 0 (transparent) and 1 (opaque).
        figsize: Figure width and height in inches. This accepts constants such
            as ``FIGSIZE_FULL``, ``FIGSIZE_WIDE``, and ``FIGSIZE_SQUARE`` from
            :mod:`tempoctrl.reproduce_figures`.
        grid: Whether to draw grid lines behind the plot.
        grid_kwargs: Optional keyword arguments passed to ``ax.grid``.
            The default matches the existing light dashed style used by
            this plotter.

    Returns:
        The Matplotlib axes containing the histogram.

    Raises:
        KeyError: If ``continuous_variable`` is not a dataframe column.
        TypeError: If the dataframe cannot be converted or the selected column
            is not numeric.
        ValueError: If a plotting option is invalid or the column has no data.

    """
    frame = _to_pandas(dataframe)

    if continuous_variable not in frame.columns:
        raise KeyError(
            f"Column {continuous_variable!r} was not found in the dataframe"
        )

    values = frame[continuous_variable]
    if pd.api.types.is_bool_dtype(
        values.dtype
    ) or not pd.api.types.is_numeric_dtype(values.dtype):
        raise TypeError(
            f"Column {continuous_variable!r} must contain numeric "
            "continuous data"
        )
    if values.notna().sum() == 0:
        raise ValueError(
            f"Column {continuous_variable!r} contains no plottable values"
        )

    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number between 0 and 1")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if (
        not isinstance(figsize, tuple)
        or len(figsize) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, float))
            or dimension <= 0
            for dimension in figsize
        )
    ):
        raise ValueError("figsize must be a tuple of two positive numbers")
    if not isinstance(grid, bool):
        raise TypeError("grid must be a boolean")
    if grid_kwargs is not None and not isinstance(grid_kwargs, dict):
        raise TypeError(
            "grid_kwargs must be a dictionary of ax.grid arguments"
        )
    if not is_color_like(histogram_color):
        raise ValueError(f"Invalid histogram_color: {histogram_color!r}")
    if not is_color_like(histogram_edgecolor):
        raise ValueError(
            f"Invalid histogram_edgecolor: {histogram_edgecolor!r}"
        )

    with plt.style.context("default"):
        _, ax = plt.subplots(figsize=figsize, dpi=100)
        sns.histplot(
            data=frame,
            x=continuous_variable,
            kde=True,
            color=histogram_color,
            edgecolor=histogram_edgecolor,
            bins=bins,
            alpha=alpha,
            ax=ax,
        )

        ax.set_axisbelow(True)
        if grid:
            if grid_kwargs is None:
                ax.grid(True)
            else:
                ax.grid(**grid_kwargs)
        else:
            ax.grid(False)
        ax.set_title(
            title
            if title is not None
            else f"Distribution of {continuous_variable}",
            fontsize=title_fontsize,
            pad=20,
        )
        ax.set_xlabel(
            continuous_variable if x_label is None else x_label,
            fontsize=12,
        )
        ax.set_ylabel("Count" if y_label is None else y_label, fontsize=12)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    return ax


def plot_bar_chart(
    dataframe: Any,
    categorical_variable: str,
    *,
    title: str | None = None,
    title_fontsize: str = "large",
    x_label: str | None = None,
    y_label: str | None = None,
    bar_color: str = "skyblue",
    bar_edgecolor: str = "white",
    alpha: float = 0.7,
    figsize: tuple[float, float] = (10.0, 6.0),
    grid: bool = True,
    grid_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """Plot alphabetically ordered counts for a categorical variable.

    The function accepts pandas dataframes, Polars dataframes, and other
    dataframe-like objects that pandas can convert. Null observations are not
    included in the category counts.

    Args:
        dataframe: Data containing the variable to plot.
        categorical_variable: Name of the categorical column to plot.
        title: Plot title. Defaults to ``"Counts of <column>"``.
        title_fontsize: Matplotlib font-size name for the title. The
            spelling is retained to match :func:`plot_histogram`.
        x_label: X-axis label. Defaults to the categorical variable's name.
        y_label: Y-axis label. Defaults to ``"Count"``.
        bar_color: Matplotlib-compatible bar color.
        bar_edgecolor: Matplotlib-compatible bar edge color.
        alpha: Bar opacity between 0 (transparent) and 1 (opaque).
        figsize: Figure width and height in inches. This accepts constants such
            as ``FIGSIZE_FULL``, ``FIGSIZE_WIDE``, and ``FIGSIZE_SQUARE`` from
            :mod:`tempoctrl.reproduce_figures`.
        grid: Whether to draw grid lines behind the plot.
        grid_kwargs: Optional keyword arguments passed to ``ax.grid``.
            The default matches the existing light dashed style used by
            this plotter.

    Returns:
        The Matplotlib axes containing the bar chart.

    Raises:
        KeyError: If ``categorical_variable`` is not a dataframe column.
        TypeError: If the dataframe cannot be converted.
        ValueError: If a plotting option is invalid or the column has no data.

    """
    frame = _to_pandas(dataframe)

    if categorical_variable not in frame.columns:
        raise KeyError(
            f"Column {categorical_variable!r} was not found in the dataframe"
        )

    values = frame[categorical_variable].dropna()
    if values.empty:
        raise ValueError(
            f"Column {categorical_variable!r} contains no categories"
        )

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number between 0 and 1")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if (
        not isinstance(figsize, tuple)
        or len(figsize) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, float))
            or dimension <= 0
            for dimension in figsize
        )
    ):
        raise ValueError("figsize must be a tuple of two positive numbers")
    if not isinstance(grid, bool):
        raise TypeError("grid must be a boolean")
    if grid_kwargs is not None and not isinstance(grid_kwargs, dict):
        raise TypeError(
            "grid_kwargs must be a dictionary of ax.grid arguments"
        )
    if not is_color_like(bar_color):
        raise ValueError(f"Invalid bar_color: {bar_color!r}")
    if not is_color_like(bar_edgecolor):
        raise ValueError(f"Invalid bar_edgecolor: {bar_edgecolor!r}")

    category_order = sorted(
        pd.unique(values),
        key=lambda category: str(category).casefold(),
    )

    with plt.style.context("default"):
        _, ax = plt.subplots(figsize=figsize, dpi=100)
        sns.countplot(
            data=frame,
            x=categorical_variable,
            order=category_order,
            color=bar_color,
            saturation=1,
            edgecolor=bar_edgecolor,
            alpha=alpha,
            ax=ax,
        )

        ax.set_axisbelow(True)
        if grid:
            if grid_kwargs is None:
                ax.grid(True)
            else:
                ax.grid(**grid_kwargs)
        else:
            ax.grid(False)
        ax.set_title(
            title
            if title is not None
            else f"Counts of {categorical_variable}",
            fontsize=title_fontsize,
            pad=20,
        )
        ax.set_xlabel(
            categorical_variable if x_label is None else x_label,
            fontsize=12,
        )
        ax.set_ylabel("Count" if y_label is None else y_label, fontsize=12)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    return ax
