"""Test summary-statistic visualizations."""

from collections.abc import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pytest
from matplotlib.axes import Axes

from tempoctrl.visualize.summary_statistics import (
    plot_bar_chart,
    plot_histogram,
)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"speed": [1.0, 2.0, 2.5, 4.0]}),
        pl.DataFrame({"speed": [1.0, 2.0, 2.5, 4.0]}),
    ],
)
def test_plot_histogram_accepts_supported_dataframes(
    frame: pd.DataFrame | pl.DataFrame,
) -> None:
    """Plot numeric values from each supported DataFrame representation.
    """
    ax = plot_histogram(
        frame,
        "speed",
        title="Player speed",
        title_fontsize="x-large",
        histogram_color="red",
        histogram_edgecolor="blue",
        bins=4,
        alpha=0.5,
    )

    assert ax.get_title() == "Player speed"
    assert ax.get_xlabel() == "speed"
    assert ax.get_ylabel() == "Count"
    assert len(ax.patches) == 4
    assert ax.title.get_fontsize() == pytest.approx(14.4)
    assert mcolors.to_rgba(ax.patches[0].get_facecolor()) == pytest.approx(
        mcolors.to_rgba("red", alpha=0.5)
    )
    assert mcolors.to_rgba(ax.patches[0].get_edgecolor()) == pytest.approx(
        mcolors.to_rgba("blue")
    )

    plt.close(ax.figure)


def test_plot_histogram_rejects_non_numeric_column() -> None:
    """Reject categorical histogram input with a clear error."""
    frame = pd.DataFrame({"position": ["left", "right"]})

    with pytest.raises(TypeError, match="numeric continuous data"):
        plot_histogram(frame, "position")


def test_plot_histogram_rejects_missing_column() -> None:
    """Reject a histogram request for an absent column."""
    frame = pd.DataFrame({"speed": [1.0]})

    with pytest.raises(KeyError, match="missing"):
        plot_histogram(frame, "missing")


def test_distribution_plots_support_grid_customization() -> None:
    """Apply explicit grid settings to numeric distribution plots."""
    frame = pd.DataFrame(
        {"speed": [1.0, 2.0, 2.5, 4.0], "position": ["A", "B", "A", "B"]}
    )

    hist_ax = plot_histogram(
        frame,
        "speed",
        grid=False,
    )
    assert not any(line.get_visible() for line in hist_ax.get_xgridlines())
    assert not any(line.get_visible() for line in hist_ax.get_ygridlines())

    custom_hist_ax = plot_histogram(
        frame,
        "speed",
        grid_kwargs={
            "color": "red",
            "linestyle": ":",
            "linewidth": 2,
            "alpha": 0.5,
        },
    )
    assert any(line.get_visible() for line in custom_hist_ax.get_xgridlines())
    assert any(line.get_visible() for line in custom_hist_ax.get_ygridlines())
    assert custom_hist_ax.xaxis.get_gridlines()[0].get_color() == "red"

    bar_ax = plot_bar_chart(
        frame,
        "position",
        grid=False,
    )
    assert not any(line.get_visible() for line in bar_ax.get_xgridlines())
    assert not any(line.get_visible() for line in bar_ax.get_ygridlines())

    custom_bar_ax = plot_bar_chart(
        frame,
        "position",
        grid_kwargs={
            "color": "green",
            "linestyle": "--",
            "linewidth": 1.5,
            "alpha": 0.75,
        },
    )
    assert any(line.get_visible() for line in custom_bar_ax.get_xgridlines())
    assert any(line.get_visible() for line in custom_bar_ax.get_ygridlines())
    assert custom_bar_ax.xaxis.get_gridlines()[0].get_color() == "green"

    plt.close(hist_ax.figure)
    plt.close(custom_hist_ax.figure)
    plt.close(bar_ax.figure)
    plt.close(custom_bar_ax.figure)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(
            {"position": ["Striker", "defender", "Goalkeeper", "defender"]}
        ),
        pl.DataFrame(
            {"position": ["Striker", "defender", "Goalkeeper", "defender"]}
        ),
    ],
)
def test_plot_bar_chart_orders_categories_alphabetically(
    frame: pd.DataFrame | pl.DataFrame,
) -> None:
    """Order shuffled category labels alphabetically in the bar chart.
    """
    ax = plot_bar_chart(
        frame,
        "position",
        title="Player positions",
        title_fontsize="x-large",
        bar_color="red",
        bar_edgecolor="blue",
        alpha=0.5,
    )

    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "defender",
        "Goalkeeper",
        "Striker",
    ]
    assert [patch.get_height() for patch in ax.patches] == [2, 1, 1]
    assert ax.get_title() == "Player positions"
    assert ax.get_xlabel() == "position"
    assert ax.get_ylabel() == "Count"
    assert ax.title.get_fontsize() == pytest.approx(14.4)
    assert mcolors.to_rgba(ax.patches[0].get_facecolor()) == pytest.approx(
        mcolors.to_rgba("red", alpha=0.5)
    )
    assert mcolors.to_rgba(ax.patches[0].get_edgecolor()) == pytest.approx(
        mcolors.to_rgba("blue", alpha=0.5)
    )

    plt.close(ax.figure)


def test_plot_bar_chart_rejects_missing_column() -> None:
    """Reject a bar-chart request for an absent category column."""
    frame = pd.DataFrame({"position": ["defender"]})

    with pytest.raises(KeyError, match="missing"):
        plot_bar_chart(frame, "missing")


@pytest.mark.parametrize(
    ("plotter", "frame", "variable"),
    [
        (plot_histogram, pd.DataFrame({"speed": [1.0, 2.0]}), "speed"),
        (
            plot_bar_chart,
            pd.DataFrame({"position": ["keeper", "winger"]}),
            "position",
        ),
    ],
)
def test_distribution_plots_use_light_default_color_scheme(
    plotter: Callable[..., Axes],
    frame: pd.DataFrame | pl.DataFrame,
    variable: str,
) -> None:
    """Use the expected default colors in both distribution plot types.
    """
    ax = plotter(frame, variable)

    assert ax.get_facecolor() == pytest.approx(mcolors.to_rgba("white"))
    assert ax.figure.get_facecolor() == pytest.approx(mcolors.to_rgba("white"))
    assert ax.patches[0].get_facecolor() == pytest.approx(
        mcolors.to_rgba("skyblue", alpha=0.7)
    )
    assert all(line.get_visible() for line in ax.get_xgridlines())
    assert all(line.get_visible() for line in ax.get_ygridlines())
    assert ax.get_axisbelow()

    plt.close(ax.figure)


@pytest.mark.parametrize(
    ("plotter", "frame", "variable", "figsize"),
    [
        (
            plot_histogram,
            pd.DataFrame({"speed": [1.0, 2.0, 3.0]}),
            "speed",
            (7.0, 4.5),
        ),
        (
            plot_bar_chart,
            pd.DataFrame({"position": ["keeper", "winger"]}),
            "position",
            (7.0, 3.5),
        ),
    ],
)
def test_distribution_plots_support_labels_figsize_and_count_formatting(
    plotter: Callable[..., Axes],
    frame: pd.DataFrame | pl.DataFrame,
    variable: str,
    figsize: tuple[float, float],
) -> None:
    """Apply explicit labels, dimensions, and count formatting to plots.
    """
    ax = plotter(
        frame,
        variable,
        x_label="Paper x label",
        y_label="Number of observations",
        figsize=figsize,
    )

    assert ax.get_xlabel() == "Paper x label"
    assert ax.get_ylabel() == "Number of observations"
    assert tuple(ax.figure.get_size_inches()) == pytest.approx(figsize)
    assert ax.yaxis.get_major_formatter()(3000) == "3,000"

    plt.close(ax.figure)


@pytest.mark.parametrize("plotter", [plot_histogram, plot_bar_chart])
def test_distribution_plots_reject_invalid_figsize(
    plotter: Callable[..., Axes],
) -> None:
    """Reject malformed dimensions instead of creating ambiguous plots.
    """
    frame = pd.DataFrame({"value": [1, 2]})

    with pytest.raises(ValueError, match="figsize"):
        plotter(frame, "value", figsize=(7.0, -1.0))
