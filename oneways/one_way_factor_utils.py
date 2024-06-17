from itertools import cycle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as ms


def get_mean_and_empirical_bounds(
        df_with_time_index: pd.DataFrame, 
        resample_freq: str,
        response_col: str,
        condition_on: str = None,
    ):
    """
    This function takes `df_with_time_index` and resamples it following `resample_freq` as 
    described in https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling as well
    as counts per resampled time.

    If `condition_on` is specified then also return all data grouped on class.

    If this is a downsample then we also return emprical 95% confidence intervals

    """
    if condition_on:
        resampled_df = df_with_time_index.groupby(condition_on)
    else:
        resampled_df = df_with_time_index.copy()
    resampled_df = resampled_df[response_col].resample(resample_freq)

    mean = resampled_df.mean()
    category_counts = resampled_df.count()
    if mean.index.nlevels == 2:
        # Here we need to make sure to get the lengths of a given category
        # instead of overall DF length
        resampled_lengths = len(mean.loc[mean.index.levels[0][0]])
    elif mean.index.nlevels == 1:
        resampled_lengths = len(mean)
    else:
        raise NotImplementedError(f"Cannot handle conditioning on more than one variable.")
        
    if resampled_lengths < df_with_time_index[response_col].shape[0]:
        lower_bound = resampled_df.quantile(0.025).fillna(mean.bfill())
        upper_bound = resampled_df.quantile(0.975).fillna(mean.bfill())
        return mean, category_counts, lower_bound, upper_bound
    else:
        return mean, category_counts, None, None


def plot_mean_and_bounds(mean: pd.Series, fig, lower_bound=None, upper_bound=None, row=1, col=1, top_n_conditions=10):
    """
    Given the results of `get_mean_and_empirical_bounds` plot the mean and bounds on `fig`
     - assigning a color to each condition.

    Drops any conditions with 0 count.

    If `top_n_conditions` is specified then only include the top n conditions by count.
    """
    def plot_bounds(lb, ub, color, fig, legendgroup, alpha=0.3):
        def hex_to_rgb(hex_color: str) -> tuple:
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = hex_color * 2
            return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        color = f"rgba{(*hex_to_rgb(color), alpha)}" if color.startswith("#") else color
        fig.add_trace(
            go.Scatter(
                x=lb.index,
                y=lb,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Lower Bound',
                legendgroup=legendgroup,
                visible="legendonly"
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=ub.index,
                y=ub,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=color,
                showlegend=False,
                name='Upper Bound',
                legendgroup=legendgroup,
                visible="legendonly"
            ),
            row=row,
            col=col,
        )

    color_gen = cycle(px.colors.qualitative.Plotly)
    color = next(color_gen)
    colors_used = []

    conditions = None
    if mean.index.nlevels == 2:
        conditions = mean.index.levels[0]
        per_condition_count = mean.groupby(level=0).count()
        conditions = per_condition_count[per_condition_count > 0].index
        if top_n_conditions and len(conditions) > top_n_conditions:
            conditions = per_condition_count.nlargest(top_n_conditions).index

        for condition in conditions:
            subset = mean.loc[condition]
            fig.add_trace(
                go.Scatter(
                    x=subset.index,
                    y=subset,
                    mode='lines',
                    name=f"{mean.name} | {condition}",
                    legendgroup=condition,
                    marker = dict(color=color),
                    visible="legendonly"
                ),
                row=row,
                col=col
            )
            colors_used.append(color)
            if lower_bound is not None:
                plot_bounds(lower_bound.loc[condition], upper_bound.loc[condition], color, fig, legendgroup=condition)
            color = next(color_gen)
    else:
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean,
                mode='lines',
                name=mean.name,
                legendgroup=mean.name,
                marker = dict(color=color),
                visible="legendonly"
            ),
            row=row,
            col=col,
        )
        # Add bounds with a fill between
        if lower_bound is not None:
            plot_bounds(lower_bound, upper_bound, color, fig, legendgroup=mean.name)
        colors_used.append(color)
        
    return fig, colors_used, conditions


def plot_value_and_counts(df, response_col, condition_on=None, freq='1W', save_dir=None):
    """
    Performs end to end one-way factor analysis on `df` with `response_col` as the response variable, conditioned on
    `condition_on` and resampled at `freq`.

    If `save` is True then the plot is saved to a file, otherwise it is displayed.
    """
    mean, counts, lower_bound, upper_bound = get_mean_and_empirical_bounds(df, freq, response_col, condition_on)
    fig = ms.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig, colors, conditions = plot_mean_and_bounds(mean, fig, lower_bound, upper_bound, row=1, col=1)
    if not condition_on:
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=counts,
                name='Counts',
                marker=dict(color=colors[0]),
                legendgroup='Value',
                visible="legendonly"
            ),
            row=2,
            col=1
        )
    else:
        for color, condition in zip(colors, conditions):
            bar = go.Bar(
                    x=counts.loc[condition].index,
                    y=counts.loc[condition],
                    name=f'Counts | {condition}',
                    legendgroup=condition,
                    marker=dict(color=color),
                    visible="legendonly"
                )
            fig.add_trace(
                bar,
                row=2,
                col=1
            )
    fig.update_layout(
        title=f"{response_col} | {condition_on}"
    )
    if not save_dir:
        fig.show()
    else:
        fig.write_html(Path(save_dir) / f"{condition_on}.html")
    return fig
