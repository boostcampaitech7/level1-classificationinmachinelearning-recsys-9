import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot_target_distribution(eda_df: pd.DataFrame) -> go.Figure:
    """
    Plot the distribution of target values.

    Args:
    eda_df (pd.DataFrame): Dataframe containing the target column.

    Returns:
    go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    _target_distribution = eda_df["target"].value_counts() / len(eda_df)
    fig.add_trace(go.Bar(x=_target_distribution.index, y=_target_distribution))
    fig.update_layout(title="Target distribution")
    return fig


def plot_barplots(bar_df: pd.DataFrame) -> go.Figure:
    """
    Plot bar plots for various features against the target.

    Args:
    bar_df (pd.DataFrame): Dataframe containing aggregated feature values.

    Returns:
    go.Figure: Plotly figure object with multiple subplots.
    """
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, subplot_titles=[
        "coinbase_premium_gap", "long_liquidations", "short_liquidations",
        "buy_volume", "sell_volume"
    ])
    fig.add_trace(go.Bar(x=bar_df["target"], y=bar_df["coinbase_premium_gap"]), row=1, col=1)
    # Add additional plots as needed
    fig.update_layout(title="Feature vs Target", showlegend=False)
    return fig
