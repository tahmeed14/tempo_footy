"""Small plotting helpers for inspecting development tracking frames."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from mplsoccer import Pitch

DEFAULT_DATA_PATH = Path("data/model/dev.parquet")
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
X_OFFSET = PITCH_LENGTH / 2
Y_OFFSET = PITCH_WIDTH / 2


def load_dev_frame(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    frame_id: int,
) -> pd.Series:
    """Load one development frame from the parquet dataset.

    ``possession_ids`` matches either development possession column, so both
    team-level and player-level possession IDs are accepted.
    """
    filters: list[tuple[str, str, Any]] = [
        ("game_id", "==", game_id),
        ("framenum", "==", frame_id),
    ]

    columns = [
        "game_id",
        "framenum",
        "dev_match_team_player_possession_id",
        "dev_match_team_possession_id",
        "balls_smooth",
        "home_players_smooth",
        "away_players_smooth",
    ]
    frame = pd.read_parquet(data_path, columns=columns, filters=filters)
    if possession_ids is not None:
        requested_ids = set(_normalise_ids(possession_ids))
        frame = frame[
            frame["dev_match_team_player_possession_id"].isin(requested_ids)
            | frame["dev_match_team_possession_id"].isin(requested_ids)
        ]
    if frame.empty:
        raise ValueError(
            "No development frame matched "
            f"game_id={game_id!r}, possession_ids={possession_ids!r}, "
            f"frame_id={frame_id!r}."
        )
    if len(frame) > 1:
        raise ValueError(
            "The requested frame matched multiple rows. Narrow "
            "possession_ids or check that frame_id is unique."
        )
    return frame.iloc[0]


def plot_dev_frame(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    frame_id: int,
    ax: plt.Axes | None = None,
    pitch_color: str = "grass",
    stripe: bool = False,
    home_color: str = "#e85d4a",
    away_color: str = "#3d8ed0",
    ball_size: float = 110,
    ball_color: str = "#f4c542",
    show_labels: bool = True,
    show_estimated: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot home players, away players, and the ball for one frame.

    Example:
    -------
    >>> from tempoctrl.gradient_sports.viztools import plot_dev_frame
    >>> plot_dev_frame(game_id=10517, possession_ids="some-id", frame_id=12345)

    """
    row = load_dev_frame(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        frame_id=frame_id,
    )
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        line_color="#f4f1e8",
        pitch_color=pitch_color,
        stripe=stripe,
        linewidth=1.5,
    )
    if ax is None:
        _, ax = pitch.draw(figsize=(12, 8))
    else:
        pitch.draw(ax=ax)
    fig = ax.figure

    _plot_players(
        ax,
        row["home_players_smooth"],
        color=home_color,
        label="Home",
        show_labels=show_labels,
        show_estimated=show_estimated,
    )
    _plot_players(
        ax,
        row["away_players_smooth"],
        color=away_color,
        label="Away",
        show_labels=show_labels,
        show_estimated=show_estimated,
    )
    _plot_ball(
        ax,
        row["balls_smooth"],
        color=ball_color,
        size=ball_size,
        show_estimated=show_estimated,
    )

    ax.set_title(f"Game {game_id} | Frame {frame_id}", fontsize=14, pad=12)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False
    )
    fig.tight_layout()
    return fig, ax


def animate_dev_frames(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    frame_ids: Iterable[int] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    interval: int = 50,
    ax: plt.Axes | None = None,
    pitch_color: str = "grass",
    stripe: bool = False,
    home_color: str = "#e85d4a",
    away_color: str = "#3d8ed0",
    ball_size: float = 110,
    ball_color: str = "#f4c542",
    show_labels: bool = True,
    show_estimated: bool = True,
) -> FuncAnimation:
    """Animate tracking data across selected frames.

    Provide either ``frame_ids`` or both ``start_frame`` and ``end_frame``.
    The returned ``FuncAnimation`` displays in a notebook when it is the last
    expression in a cell, or can be rendered with ``animation.to_jshtml()``.
    """
    if frame_ids is not None and (
        start_frame is not None or end_frame is not None
    ):
        raise ValueError("Use frame_ids or start_frame/end_frame, not both.")
    if frame_ids is None and (start_frame is None or end_frame is None):
        raise ValueError(
            "Provide frame_ids or both start_frame and end_frame."
        )
    if (
        start_frame is not None
        and end_frame is not None
        and start_frame > end_frame
    ):
        raise ValueError(
            "start_frame must be less than or equal to end_frame."
        )

    frames = _load_dev_frames(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        frame_ids=frame_ids,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        line_color="#f4f1e8",
        pitch_color=pitch_color,
        stripe=stripe,
        linewidth=1.5,
    )
    if ax is None:
        _, ax = pitch.draw(figsize=(12, 8))
    else:
        pitch.draw(ax=ax)
    fig = ax.figure

    home_scatter = ax.scatter(
        [],
        [],
        s=360,
        c=home_color,
        edgecolors="#fffdf5",
        linewidths=1.2,
        label="Home",
        zorder=3,
    )
    away_scatter = ax.scatter(
        [],
        [],
        s=360,
        c=away_color,
        edgecolors="#fffdf5",
        linewidths=1.2,
        label="Away",
        zorder=3,
    )
    ball_scatter = ax.scatter(
        [],
        [],
        s=ball_size,
        c=ball_color,
        edgecolors="#20251f",
        linewidths=1.5,
        label="Ball",
        zorder=5,
    )
    home_labels = _make_animation_labels(
        ax, frames["home_players_smooth"], show_labels
    )
    away_labels = _make_animation_labels(
        ax, frames["away_players_smooth"], show_labels
    )

    def update(frame_index: int) -> tuple[Any, ...]:
        row = frames.iloc[frame_index]
        artists: list[Any] = [home_scatter, away_scatter, ball_scatter]
        artists.extend(
            _update_players(
                home_scatter,
                home_labels,
                row["home_players_smooth"],
                show_estimated,
            )
        )
        artists.extend(
            _update_players(
                away_scatter,
                away_labels,
                row["away_players_smooth"],
                show_estimated,
            )
        )
        ball = row["balls_smooth"]
        if (
            ball
            and ball.get("x") is not None
            and ball.get("y") is not None
            and (show_estimated or ball.get("visibility") == "VISIBLE")
        ):
            ball_scatter.set_offsets(
                [[ball["x"] + X_OFFSET, ball["y"] + Y_OFFSET]]
            )
        else:
            ball_scatter.set_offsets([])
        ax.set_title(
            f"Game {game_id} | Frame {int(row['framenum'])}",
            fontsize=14,
            pad=12,
        )
        return tuple(artists)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False
    )
    fig.tight_layout()
    return FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=False,
        repeat=True,
    )


def _load_dev_frames(
    data_path: str | Path,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None,
    frame_ids: Iterable[int] | None,
    start_frame: int | None,
    end_frame: int | None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, Any]] = [("game_id", "==", game_id)]
    if start_frame is not None:
        filters.append(("framenum", ">=", start_frame))
    if end_frame is not None:
        filters.append(("framenum", "<=", end_frame))
    columns = [
        "game_id",
        "framenum",
        "dev_match_team_player_possession_id",
        "dev_match_team_possession_id",
        "balls_smooth",
        "home_players_smooth",
        "away_players_smooth",
    ]
    frames = pd.read_parquet(data_path, columns=columns, filters=filters)
    if possession_ids is not None:
        requested_ids = set(_normalise_ids(possession_ids))
        frames = frames[
            frames["dev_match_team_player_possession_id"].isin(requested_ids)
            | frames["dev_match_team_possession_id"].isin(requested_ids)
        ]
    if frame_ids is not None:
        frames = frames[frames["framenum"].isin(list(frame_ids))]
    frames = frames.sort_values("framenum").reset_index(drop=True)
    if frames.empty:
        raise ValueError(
            "No development frames matched the requested selection."
        )
    return frames


def _make_animation_labels(
    ax: plt.Axes,
    player_frames: Iterable[list[dict[str, Any]] | None],
    show_labels: bool,
) -> list[plt.Text]:
    if not show_labels:
        return []
    max_players = max(
        (
            len(players) if players is not None else 0
            for players in player_frames
        ),
        default=0,
    )
    return [
        ax.text(
            0,
            0,
            "",
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )
        for _ in range(max_players)
    ]


def _update_players(
    scatter: Any,
    labels: list[plt.Text],
    players: list[dict[str, Any]] | None,
    show_estimated: bool,
) -> list[plt.Artist]:
    visible_players = [
        player
        for player in (players if players is not None else [])
        if show_estimated or player.get("visibility") == "VISIBLE"
    ]
    scatter.set_offsets(
        [
            [player["x"] + X_OFFSET, player["y"] + Y_OFFSET]
            for player in visible_players
        ]
        if visible_players
        else []
    )
    for index, label in enumerate(labels):
        if index < len(visible_players):
            player = visible_players[index]
            label.set_position(
                (player["x"] + X_OFFSET, player["y"] + Y_OFFSET)
            )
            label.set_text(str(player.get("jerseyNum", "")))
            label.set_visible(True)
        else:
            label.set_visible(False)
    return labels


def _normalise_ids(possession_ids: str | Iterable[str]) -> list[str]:
    if isinstance(possession_ids, str):
        return [possession_ids]
    return list(possession_ids)


def _plot_players(
    ax: plt.Axes,
    players: list[dict[str, Any]] | None,
    *,
    color: str,
    label: str,
    show_labels: bool,
    show_estimated: bool,
) -> None:
    if players is None or len(players) == 0:
        return
    visible_players = [
        player
        for player in players
        if show_estimated or player.get("visibility") == "VISIBLE"
    ]
    if not visible_players:
        return
    x_values = [player["x"] + X_OFFSET for player in visible_players]
    y_values = [player["y"] + Y_OFFSET for player in visible_players]
    ax.scatter(
        x_values,
        y_values,
        s=360,
        c=color,
        edgecolors="#fffdf5",
        linewidths=1.2,
        alpha=0.9,
        label=label,
        zorder=3,
    )
    if show_labels:
        for player in visible_players:
            ax.text(
                player["x"] + X_OFFSET,
                player["y"] + Y_OFFSET,
                str(player.get("jerseyNum", "")),
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=4,
            )


def _plot_ball(
    ax: plt.Axes,
    ball: dict[str, Any] | None,
    *,
    color: str,
    size: float,
    show_estimated: bool,
) -> None:
    if not ball or ball.get("x") is None or ball.get("y") is None:
        return
    if not show_estimated and ball.get("visibility") != "VISIBLE":
        return
    ax.scatter(
        ball["x"] + X_OFFSET,
        ball["y"] + Y_OFFSET,
        s=size,
        c=color,
        edgecolors="#20251f",
        linewidths=1.5,
        marker="o",
        label="Ball",
        zorder=5,
    )
