import numpy as np

import plotly.graph_objects as go


def fill_between(x, y_lower, y_upper, color, **kwargs):
    h = go.Scatter(
        x=np.hstack([x, x[::-1]]),
        y=np.hstack([y_upper, y_lower[::-1]]),
        fill="toself",
        fillcolor=color,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        **kwargs
    )
    return h
