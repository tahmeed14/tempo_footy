"""Small plotting helpers for inspecting development tracking frames."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mplsoccer import Pitch

DEFAULT_DATA_PATH = Path("data/analysis/possessions.parquet")
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
    figsize: tuple[float, float] = (12.0, 8.0),
    pitch_color: str = "grass",
    stripe: bool = False,
    home_color: str = "#e85d4a",
    away_color: str = "#3d8ed0",
    ball_size: float = 110,
    ball_color: str = "#f4c542",
    show_labels: bool = True,
    show_estimated: bool = True,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    show_home: bool = True,
    show_away: bool = True,
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
        _, ax = pitch.draw(figsize=figsize)
    else:
        pitch.draw(ax=ax)
    fig = ax.figure

    if show_home:
        _plot_players(
            ax,
            row["home_players_smooth"],
            color=home_color,
            label=home_team_name,
            show_labels=show_labels,
            show_estimated=show_estimated,
        )
    if show_away:
        _plot_players(
            ax,
            row["away_players_smooth"],
            color=away_color,
            label=away_team_name,
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


def plot_dev_start_frame(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    frame_id: int,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    title: str | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot all home and away players at the selected starting frame."""
    fig, ax = plot_dev_frame(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        frame_id=frame_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        **kwargs,
    )
    ax.set_title(
        title
        if title is not None
        else f"{home_team_name} vs {away_team_name} | Frame {frame_id}",
        fontsize=14,
        pad=12,
    )
    return fig, ax


def plot_dev_players_to_pass(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    start_frame: int,
    pass_frame: int | None = None,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    title: str | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot every player's movement from the start to the pass attempt."""
    if pass_frame is None:
        pass_frame = _resolve_pa_frame(
            data_path,
            game_id=game_id,
            possession_ids=possession_ids,
            start_frame=start_frame,
            end_frame=None,
        )
    fig, ax = plot_dev_movement(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        start_frame=start_frame,
        end_frame=pass_frame,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        **kwargs,
    )
    ax.set_title(
        title
        if title is not None
        else f"{home_team_name} vs {away_team_name} | Start to Pass Attempt",
        fontsize=14,
        pad=12,
    )
    return fig, ax


def plot_dev_possession_movement(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    pass_frame: int | None = None,
    end_frame: int | None = None,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    ball_trajectory_color: str = "darkorange",
    title: str | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot movement and trajectory from a pass to possession end."""
    if pass_frame is None:
        pass_frame = _resolve_pa_frame(
            data_path,
            game_id=game_id,
            possession_ids=possession_ids,
            start_frame=None,
            end_frame=end_frame,
        )
    if end_frame is None:
        end_frame = _resolve_last_frame(
            data_path,
            game_id=game_id,
            possession_ids=possession_ids,
        )
    fig, ax = plot_dev_movement(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        start_frame=pass_frame,
        end_frame=end_frame,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        ball_trajectory_color=ball_trajectory_color,
        show_ball_trajectory=True,
        **kwargs,
    )
    ax.set_title(
        title
        if title is not None
        else (
            f"{home_team_name} vs {away_team_name} | "
            "Pass Attempt to Possession End"
        ),
        fontsize=14,
        pad=12,
    )
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
    figsize: tuple[float, float] = (12.0, 8.0),
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
        _, ax = pitch.draw(figsize=figsize)
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


def plot_dev_movement(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    frame_ids: Iterable[int] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    interval: int = 50,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
    pitch_color: str = "grass",
    stripe: bool = False,
    home_color: str = "#e85d4a",
    away_color: str = "#3d8ed0",
    ball_size: float = 110,
    ball_color: str = "#f4c542",
    ball_start_color: str = "#2ca25f",
    ball_end_color: str = "#de2d26",
    start_player_size: float = 90,
    show_labels: bool = True,
    show_estimated: bool = True,
    show_home: bool = True,
    show_away: bool = True,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    show_ball_trajectory: bool = False,
    ball_trajectory_color: str = "darkorange",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot player movement from the first selected frame to the last.

    ``frame_ids`` or both ``start_frame`` and ``end_frame`` select the frame
    range, using the same rules as :func:`animate_dev_frames`. Players are
    matched between endpoints by jersey number. The returned figure shows
    smaller start markers, normal-sized endpoint markers, and dashed arrows.
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
        _, ax = pitch.draw(figsize=figsize)
    else:
        pitch.draw(ax=ax)
    fig = ax.figure

    first_frame = frames.iloc[0]
    last_frame = frames.iloc[-1]
    if show_home:
        _plot_player_movement(
            ax,
            first_frame["home_players_smooth"],
            last_frame["home_players_smooth"],
            color=home_color,
            start_size=start_player_size,
            end_size=360,
            show_labels=show_labels,
            show_estimated=show_estimated,
        )
    if show_away:
        _plot_player_movement(
            ax,
            first_frame["away_players_smooth"],
            last_frame["away_players_smooth"],
            color=away_color,
            start_size=start_player_size,
            end_size=360,
            show_labels=show_labels,
            show_estimated=show_estimated,
        )
    _plot_endpoint_ball(
        ax,
        first_frame["balls_smooth"],
        ball_start_color,
        ball_size,
        show_estimated,
    )
    _plot_endpoint_ball(
        ax,
        last_frame["balls_smooth"],
        ball_end_color,
        ball_size,
        show_estimated,
    )
    if show_ball_trajectory:
        ball_positions = _ball_positions(frames, show_estimated)
        _plot_ball_segment(ax, ball_positions, ball_trajectory_color)

    movement_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ball_start_color,
            markeredgecolor="#20251f",
            label="Ball start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ball_end_color,
            markeredgecolor="#20251f",
            label="Ball end",
        ),
    ]
    if show_ball_trajectory:
        movement_handles.insert(
            0,
            Line2D(
                [0],
                [0],
                color=ball_trajectory_color,
                linestyle="--",
                marker=">",
                label="Ball movement",
            ),
        )
    if show_home:
        movement_handles.insert(
            0,
            Line2D(
                [0],
                [0],
                color=home_color,
                linestyle="--",
                marker=">",
                label=f"{home_team_name} movement",
            ),
        )
    if show_away:
        movement_handles.insert(
            1 if show_home else 0,
            Line2D(
                [0],
                [0],
                color=away_color,
                linestyle="--",
                marker=">",
                label=f"{away_team_name} movement",
            ),
        )
    ax.legend(
        handles=movement_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
    )
    ax.set_title(
        f"Game {game_id} | Frames "
        f"{int(first_frame['framenum'])}-"
        f"{int(last_frame['framenum'])}",
        fontsize=14,
        pad=12,
    )
    fig.tight_layout()
    return fig, ax


def plot_dev_player_movement(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None = None,
    frame_ids: Iterable[int] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    interval: int = 50,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
    pitch_color: str = "grass",
    stripe: bool = False,
    home_color: str = "#e85d4a",
    away_color: str = "#3d8ed0",
    ball_size: float = 110,
    ball_color: str = "#f4c542",
    ball_start_color: str = "#2ca25f",
    ball_end_color: str = "#de2d26",
    pa_trajectory_color: str = "#cc6b1f",
    start_player_size: float = 90,
    end_player_size: float = 360,
    pa_player_size: float = 220,
    ball_trajectory_size: float = 24,
    pass_attempt_color: str = "#20251f",
    title: str | None = None,
    show_start_player: bool = True,
    show_end_player: bool = False,
    show_pa_player: bool = True,
    show_labels: bool = True,
    show_estimated: bool = True,
    show_home: bool = True,
    show_away: bool = True,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    home_jersey: str | int | None = None,
    away_jersey: str | int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one player's movement, endpoint positions, and ball movement.

    Exactly one of ``home_jersey`` or ``away_jersey`` selects the player whose
    complete trajectory is shown. Other visible teams are plotted only at the
    final frame. If a ``PA`` row exists in the selected range, the ball path
    before that frame is white and the remaining path uses
    ``pa_trajectory_color``.
    """
    if (home_jersey is None) == (away_jersey is None):
        raise ValueError("Provide exactly one of home_jersey or away_jersey.")
    if home_jersey is not None and not show_home:
        raise ValueError(
            "home_jersey cannot be selected when show_home=False."
        )
    if away_jersey is not None and not show_away:
        raise ValueError(
            "away_jersey cannot be selected when show_away=False."
        )

    frames = _load_dev_frames(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        frame_ids=frame_ids,
        start_frame=start_frame,
        end_frame=end_frame,
        include_event_type=True,
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
        _, ax = pitch.draw(figsize=figsize)
    else:
        pitch.draw(ax=ax)
    fig = ax.figure
    first_frame = frames.iloc[0]
    last_frame = frames.iloc[-1]
    selected_color = home_color if home_jersey is not None else away_color
    selected_team_name = (
        home_team_name if home_jersey is not None else away_team_name
    )
    selected_jersey = home_jersey if home_jersey is not None else away_jersey

    if show_home:
        _plot_endpoint_players(
            ax,
            last_frame["home_players_smooth"],
            home_color,
            show_labels,
            show_estimated,
            exclude_jersey=home_jersey,
        )
    if show_away:
        _plot_endpoint_players(
            ax,
            last_frame["away_players_smooth"],
            away_color,
            show_labels,
            show_estimated,
            exclude_jersey=away_jersey,
        )
    _plot_selected_player_trajectory(
        ax,
        frames,
        "home_players_smooth"
        if home_jersey is not None
        else "away_players_smooth",
        selected_jersey,
        selected_color,
        start_player_size,
        end_player_size,
        pa_player_size,
        show_labels,
        show_estimated,
        pa_frame=_first_pa_frame(frames),
        show_start_player=show_start_player,
        show_end_player=show_end_player,
        show_pa_player=show_pa_player,
    )
    _plot_ball_trajectory(
        ax,
        frames,
        ball_size=ball_size,
        ball_color=ball_color,
        ball_start_color=ball_start_color,
        ball_end_color=ball_end_color,
        pa_trajectory_color=pa_trajectory_color,
        ball_trajectory_size=ball_trajectory_size,
        pass_attempt_color=pass_attempt_color,
        show_estimated=show_estimated,
    )

    pa_frame = _first_pa_frame(frames)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=selected_color,
            marker="o",
            label=f"{selected_team_name} #{selected_jersey}",
        ),
        Line2D(
            [0],
            [0],
            color=selected_color,
            marker="o",
            markersize=5,
            label="Other players: final frame",
        ),
        Line2D(
            [0],
            [0],
            color="white",
            linestyle="--",
            label="Ball trajectory before PA",
        ),
        Line2D(
            [0],
            [0],
            color=pa_trajectory_color,
            linestyle="--",
            label="Ball trajectory from PA",
        ),
        Line2D(
            [0],
            [0],
            color=pa_trajectory_color,
            marker="o",
            markersize=4,
            label="Ball position after PA",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ball_start_color,
            markeredgecolor="#20251f",
            label="Ball start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ball_end_color,
            markeredgecolor="#20251f",
            label="Ball end",
        ),
    ]
    if show_end_player:
        legend_handles.insert(
            1,
            Line2D(
                [0],
                [0],
                color=selected_color,
                marker="o",
                markersize=8,
                label="Selected player end",
            ),
        )
    if show_start_player:
        legend_handles.insert(
            1,
            Line2D(
                [0],
                [0],
                color=selected_color,
                marker="o",
                markersize=5,
                label="Selected player start",
            ),
        )
    if show_pa_player and pa_frame is not None:
        legend_handles.insert(
            3,
            Line2D(
                [0],
                [0],
                color=selected_color,
                marker="o",
                markersize=7,
                label="Selected player at PA",
            ),
        )
    if pa_frame is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=pass_attempt_color,
                marker="x",
                markersize=5,
                label="Pass Attempted",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        frameon=False,
    )
    ax.set_title(
        title
        if title is not None
        else (
            f"Game {game_id} | Frames "
            f"{int(first_frame['framenum'])}-"
            f"{int(last_frame['framenum'])}"
        ),
        fontsize=14,
        pad=12,
    )
    fig.tight_layout()
    return fig, ax


def _plot_player_movement(
    ax: plt.Axes,
    start_players: list[dict[str, Any]] | None,
    end_players: list[dict[str, Any]] | None,
    *,
    color: str,
    start_size: float,
    end_size: float,
    show_labels: bool,
    show_estimated: bool,
) -> None:
    start_by_jersey = _players_by_jersey(start_players, show_estimated)
    end_by_jersey = _players_by_jersey(end_players, show_estimated)

    for jersey, start_player in start_by_jersey.items():
        end_player = end_by_jersey.get(jersey)
        if end_player is None:
            continue
        ax.annotate(
            "",
            xy=_player_position(end_player),
            xytext=_player_position(start_player),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linestyle": "--",
                "linewidth": 1.2,
                "alpha": 0.8,
                "shrinkA": 5,
                "shrinkB": 8,
            },
            zorder=2,
        )

    if start_by_jersey:
        start_positions = [
            _player_position(player) for player in start_by_jersey.values()
        ]
        ax.scatter(
            [position[0] for position in start_positions],
            [position[1] for position in start_positions],
            s=start_size,
            c=color,
            edgecolors="#fffdf5",
            linewidths=1.0,
            alpha=0.55,
            zorder=3,
        )
    if end_by_jersey:
        end_positions = [
            _player_position(player) for player in end_by_jersey.values()
        ]
        ax.scatter(
            [position[0] for position in end_positions],
            [position[1] for position in end_positions],
            s=end_size,
            c=color,
            edgecolors="#fffdf5",
            linewidths=1.2,
            alpha=0.9,
            zorder=4,
        )
        if show_labels:
            for player in end_by_jersey.values():
                ax.text(
                    *_player_position(player),
                    str(player.get("jerseyNum", "")),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    zorder=5,
                )


def _plot_endpoint_ball(
    ax: plt.Axes,
    ball: dict[str, Any] | None,
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
        zorder=6,
    )


def _plot_endpoint_players(
    ax: plt.Axes,
    players: list[dict[str, Any]] | None,
    color: str,
    show_labels: bool,
    show_estimated: bool,
    *,
    exclude_jersey: str | int | None = None,
) -> None:
    players_by_jersey = _players_by_jersey(players, show_estimated)
    if exclude_jersey is not None:
        players_by_jersey.pop(str(exclude_jersey), None)
    if not players_by_jersey:
        return
    positions = [
        _player_position(player) for player in players_by_jersey.values()
    ]
    ax.scatter(
        [position[0] for position in positions],
        [position[1] for position in positions],
        s=360,
        c=color,
        edgecolors="#fffdf5",
        linewidths=1.2,
        alpha=0.9,
        zorder=4,
    )
    if show_labels:
        for player in players_by_jersey.values():
            ax.text(
                *_player_position(player),
                str(player.get("jerseyNum", "")),
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=5,
            )


def _plot_selected_player_trajectory(
    ax: plt.Axes,
    frames: pd.DataFrame,
    player_column: str,
    jersey: str | int | None,
    color: str,
    start_size: float,
    end_size: float,
    pa_size: float,
    show_labels: bool,
    show_estimated: bool,
    pa_frame: int | None,
    show_start_player: bool,
    show_end_player: bool,
    show_pa_player: bool,
) -> None:
    target_jersey = str(jersey)
    positions: list[tuple[int, tuple[float, float]]] = []
    final_player: dict[str, Any] | None = None
    for _, row in frames.iterrows():
        players = row[player_column]
        matching_players = _players_by_jersey(players, show_estimated)
        player = matching_players.get(target_jersey)
        if player is not None:
            positions.append((int(row["framenum"]), _player_position(player)))
            final_player = player
    if not positions or final_player is None:
        raise ValueError(
            f"No visible player matched jersey {jersey!r} in the "
            "selected frames."
        )
    if show_start_player:
        ax.scatter(
            *positions[0][1],
            s=max(start_size, 2.5 * 110),
            facecolors="none",
            edgecolors=color,
            linewidths=2.0,
            alpha=0.95,
            zorder=3,
        )
    if show_end_player:
        ax.scatter(
            *positions[-1][1],
            s=end_size,
            c=color,
            edgecolors="#fffdf5",
            linewidths=1.2,
            alpha=0.9,
            zorder=4,
        )
    if show_pa_player and pa_frame is not None:
        pa_position = next(
            (position for frame, position in positions if frame == pa_frame),
            None,
        )
        if pa_position is not None:
            ax.scatter(
                *pa_position,
                s=max(pa_size, 1.5 * 110),
                facecolors="none",
                edgecolors=color,
                linewidths=2.0,
                alpha=0.95,
                zorder=5,
            )
    if show_labels:
        label_size = max(6.0, 8.0 * (end_size / 360.0) ** 0.5)
        if show_end_player:
            ax.text(
                *positions[-1][1],
                str(final_player.get("jerseyNum", jersey)),
                color="white",
                ha="center",
                va="center",
                fontsize=label_size,
                fontweight="bold",
                zorder=6,
            )


def _plot_ball_trajectory(
    ax: plt.Axes,
    frames: pd.DataFrame,
    *,
    ball_size: float,
    ball_color: str,
    ball_start_color: str,
    ball_end_color: str,
    pa_trajectory_color: str,
    ball_trajectory_size: float,
    pass_attempt_color: str,
    show_estimated: bool,
) -> None:
    ball_positions = []
    for _, row in frames.iterrows():
        ball = row["balls_smooth"]
        if (
            ball
            and ball.get("x") is not None
            and ball.get("y") is not None
            and (show_estimated or ball.get("visibility") == "VISIBLE")
        ):
            ball_positions.append(
                (int(row["framenum"]), _player_position(ball))
            )
    if not ball_positions:
        return
    pa_frames = frames.loc[frames["possession_event_type"] == "PA", "framenum"]
    pa_frame = int(pa_frames.iloc[0]) if not pa_frames.empty else None
    if len(ball_positions) > 1:
        split_index = next(
            (
                index
                for index, (frame, _) in enumerate(ball_positions)
                if pa_frame is not None and frame >= pa_frame
            ),
            len(ball_positions) - 1,
        )
        _plot_ball_segment(
            ax,
            ball_positions[: split_index + 1],
            ball_color if pa_frame is None else "white",
        )
        if pa_frame is not None and split_index < len(ball_positions) - 1:
            _plot_ball_segment(
                ax, ball_positions[split_index:], pa_trajectory_color
            )
            post_pass = ball_positions[split_index:]
            ax.scatter(
                [position[1][0] for position in post_pass],
                [position[1][1] for position in post_pass],
                s=ball_trajectory_size,
                c=pa_trajectory_color,
                edgecolors="none",
                zorder=8,
            )
    ax.scatter(
        *ball_positions[0][1],
        s=ball_size,
        c=ball_start_color,
        edgecolors="#20251f",
        linewidths=1.5,
        zorder=9,
    )
    ax.scatter(
        *ball_positions[-1][1],
        s=ball_size,
        c=ball_end_color,
        edgecolors="#20251f",
        linewidths=1.5,
        zorder=9,
    )
    if pa_frame is not None:
        pa_position = next(
            (
                position
                for frame, position in ball_positions
                if frame == pa_frame
            ),
            None,
        )
        if pa_position is not None:
            ax.scatter(
                *pa_position,
                s=ball_trajectory_size * 1.8,
                c=pass_attempt_color,
                marker="x",
                linewidths=1.2,
                zorder=10,
            )
            ax.annotate(
                "Pass Attempted",
                xy=pa_position,
                xytext=(5, 5),
                textcoords="offset points",
                color=pass_attempt_color,
                fontsize=8,
                zorder=10,
            )


def _plot_ball_segment(
    ax: plt.Axes, positions: list[tuple[int, tuple[float, float]]], color: str
) -> None:
    if len(positions) < 2:
        return
    coordinates = [position for _, position in positions]
    ax.plot(
        [position[0] for position in coordinates],
        [position[1] for position in coordinates],
        color=color,
        linestyle="--",
        linewidth=1.8,
        zorder=2,
    )
    ax.annotate(
        "",
        xy=coordinates[-1],
        xytext=coordinates[-2],
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linestyle": "--",
            "linewidth": 1.8,
        },
        zorder=3,
    )


def _first_pa_frame(frames: pd.DataFrame) -> int | None:
    pa_frames = frames.loc[frames["possession_event_type"] == "PA", "framenum"]
    return int(pa_frames.iloc[0]) if not pa_frames.empty else None


def _resolve_pa_frame(
    data_path: str | Path,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None,
    start_frame: int | None,
    end_frame: int | None,
) -> int:
    frames = _load_dev_frames(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        frame_ids=None,
        start_frame=start_frame,
        end_frame=end_frame,
        include_event_type=True,
    )
    pa_frame = _first_pa_frame(frames)
    if pa_frame is None:
        raise ValueError(
            "No PA possession event was found in the selected possession."
        )
    return pa_frame


def _resolve_last_frame(
    data_path: str | Path,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None,
) -> int:
    frames = _load_dev_frames(
        data_path,
        game_id=game_id,
        possession_ids=possession_ids,
        frame_ids=None,
        start_frame=None,
        end_frame=None,
    )
    return int(frames["framenum"].max())


def _ball_positions(
    frames: pd.DataFrame,
    show_estimated: bool,
) -> list[tuple[int, tuple[float, float]]]:
    positions = []
    for _, row in frames.iterrows():
        ball = row["balls_smooth"]
        if (
            ball
            and ball.get("x") is not None
            and ball.get("y") is not None
            and (show_estimated or ball.get("visibility") == "VISIBLE")
        ):
            positions.append((int(row["framenum"]), _player_position(ball)))
    return positions


def _players_by_jersey(
    players: list[dict[str, Any]] | None,
    show_estimated: bool,
) -> dict[Any, dict[str, Any]]:
    return {
        str(player.get("jerseyNum")): player
        for player in (players if players is not None else [])
        if player.get("jerseyNum") is not None
        and player.get("x") is not None
        and player.get("y") is not None
        and (show_estimated or player.get("visibility") == "VISIBLE")
    }


def _player_position(player: dict[str, Any]) -> tuple[float, float]:
    return player["x"] + X_OFFSET, player["y"] + Y_OFFSET


def _load_dev_frames(
    data_path: str | Path,
    *,
    game_id: int,
    possession_ids: str | Iterable[str] | None,
    frame_ids: Iterable[int] | None,
    start_frame: int | None,
    end_frame: int | None,
    include_event_type: bool = False,
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
    if include_event_type:
        columns.append("possession_event_type")
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
