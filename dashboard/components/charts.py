"""
Reusable chart components — consistent Plotly styling for Portfolio OS.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Standard theme ───────────────────────────────────────────────────────────

DARK_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

PALETTE = {
    "primary": "#00d4aa",
    "secondary": "#6c63ff",
    "warning": "#ff9f43",
    "danger": "#ff6b6b",
    "accent": "#ee5a24",
    "muted": "#888888",
}


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    name: str = "",
    color: str = PALETTE["primary"],
    fill: bool = False,
    height: int = 350,
    yaxis_title: str = "",
) -> go.Figure:
    """Standard single-series line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y],
        mode="lines", name=name or y,
        line=dict(color=color, width=2),
        fill="tozeroy" if fill else "none",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.08)" if fill else None,
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title=yaxis_title, **DARK_LAYOUT,
    )
    return fig


def multi_line_chart(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    names: list[str] | None = None,
    colors: list[str] | None = None,
    height: int = 350,
    yaxis_title: str = "",
) -> go.Figure:
    """Multi-series line chart."""
    fig = go.Figure()
    names = names or y_cols
    colors = colors or list(PALETTE.values())
    for i, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[col],
                mode="lines", name=names[i],
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title=yaxis_title, **DARK_LAYOUT,
    )
    return fig


def bar_chart(
    x: list,
    y: list,
    color: str = PALETTE["primary"],
    height: int = 300,
    yaxis_title: str = "",
) -> go.Figure:
    """Simple bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, marker_color=color))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title=yaxis_title, **DARK_LAYOUT,
    )
    return fig


def grouped_bar_chart(
    x: list,
    series: dict[str, list],
    colors: list[str] | None = None,
    height: int = 400,
    yaxis_title: str = "",
) -> go.Figure:
    """Grouped bar chart with multiple series."""
    fig = go.Figure()
    colors = colors or list(PALETTE.values())
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            name=name, x=x, y=y,
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        barmode="group", height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title=yaxis_title, **DARK_LAYOUT,
    )
    return fig


def pie_chart(
    labels: list,
    values: list,
    colors: list[str] | None = None,
    height: int = 300,
) -> go.Figure:
    """Donut-style pie chart."""
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels, values=values,
        marker_colors=colors or list(PALETTE.values()),
        textposition="inside", textinfo="label+percent",
        hole=0.35,
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        **DARK_LAYOUT, showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def heatmap(
    z: list | pd.DataFrame,
    x: list,
    y: list,
    colorscale: str = "RdBu_r",
    height: int = 400,
    zmin: float = -1,
    zmax: float = 1,
) -> go.Figure:
    """Annotated heatmap (e.g., correlation matrix)."""
    fig = px.imshow(
        z, x=x, y=y,
        color_continuous_scale=colorscale,
        zmin=zmin, zmax=zmax, text_auto=".2f",
    )
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        **DARK_LAYOUT,
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B'."""
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))
