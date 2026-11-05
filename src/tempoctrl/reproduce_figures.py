"""Reproduce figures used in the project paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from tempoctrl.reproduce_distribution_visuals import (
    DEFAULT_ALPHA_PLAYER_FIGURE,
    DEFAULT_MU_PLAYER_FIGURE,
    DEFAULT_POSTERIOR_PATH,
    plot_top_bottom_alpha_player_effects,
    plot_top_bottom_mu_player_effects,
)
from tempoctrl.reproduce_tables import MODEL_DATA_PATH
from tempoctrl.visualize.possessions import (
    plot_dev_players_to_pass,
    plot_dev_possession_movement,
    plot_dev_start_frame,
)
from tempoctrl.visualize.summary_statistics import plot_histogram

FIGSIZE_FULL = (7.0, 4.5)
FIGSIZE_WIDE = (7.0, 3.5)
FIGSIZE_SQUARE = (5.5, 5.5)
POSS_EXAMPLE = "src/tempoctrl/possession_example.parquet"
FIGURES_PATH = "paper/figures"


def save_quarto_figure(
    fig: Figure,
    filename: str,
    output_dir: str = FIGURES_PATH,
    dpi: int = 300,
) -> None:
    """Save a figure in PDF and PNG formats for Quarto."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output_dir / filename

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )


def ball_speed_tempo_histograms() -> None:
    """Save the ball-speed tempo distribution at each paper figure size."""
    model_df = pl.read_parquet(
        MODEL_DATA_PATH,
        columns=["ball_speed_tempo_player"],
    )
    figure_sizes = {
        "full": FIGSIZE_FULL,
        "wide": FIGSIZE_WIDE,
        "square": FIGSIZE_SQUARE,
    }

    for size_name, figsize in figure_sizes.items():
        ax = plot_histogram(
            model_df,
            continuous_variable="ball_speed_tempo_player",
            title="Distribution of Ball Speed Tempo",
            y_label="# of Possessions",
            x_label="Tempo Speed (metres/second)",
            figsize=figsize,
        )
        save_quarto_figure(
            ax.figure,
            filename=f"ball_speed_tempo_distribution_{size_name}",
        )
        plt.close(ax.figure)


def possession_example() -> None:
    """Save figures illustrating an example possession."""
    start_frame, start_ax = plot_dev_start_frame(
        data_path=POSS_EXAMPLE,
        game_id=10517,
        possession_ids="10517_Argentina_1531_403",
        frame_id=65972,
        home_team_name="Argentina",
        away_team_name="France",
        home_color="skyblue",
        away_color="darkblue",
        ball_color="white",
        title="Argentina vs France | Player Possession Start",
        # figsize = FIGSIZE_SQUARE
    )

    pass_fig, pass_ax = plot_dev_players_to_pass(
        data_path=POSS_EXAMPLE,
        game_id=10517,
        possession_ids="10517_Argentina_1531_403",
        start_frame=65972,
        home_team_name="Argentina",
        away_team_name="France",
        home_color="skyblue",
        away_color="darkblue",
        ball_color="white",
        ball_start_color="white",
        ball_end_color="gray",
        start_player_size=55,
        title="Argentina vs France | Player Attempts Pass",
        # figsize = FIGSIZE_SQUARE
    )

    possession_fig, possession_ax = plot_dev_possession_movement(
        data_path=POSS_EXAMPLE,
        game_id=10517,
        possession_ids="10517_Argentina_1531_403",
        home_team_name="Argentina",
        away_team_name="France",
        home_color="skyblue",
        away_color="darkblue",
        ball_start_color="white",
        ball_end_color="gray",
        ball_trajectory_color="darkorange",
        title="Argentina vs France | Pass Successful to Teammate",
        # figsize = FIGSIZE_FULL
    )

    save_quarto_figure(start_frame, filename="player_possession_start")
    save_quarto_figure(pass_fig, filename="player_possession_pass_attempt")
    save_quarto_figure(
        possession_fig, filename="player_possession_pass_success"
    )


def main() -> None:
    """Reproduce all paper figures."""
    possession_example()
    ball_speed_tempo_histograms()

    plot_top_bottom_mu_player_effects(
        DEFAULT_POSTERIOR_PATH,
        output_path=DEFAULT_MU_PLAYER_FIGURE,
    )
    plot_top_bottom_alpha_player_effects(
        DEFAULT_POSTERIOR_PATH,
        output_path=DEFAULT_ALPHA_PLAYER_FIGURE,
    )


if __name__ == "__main__":
    main()
