import funcnodes as fn
from lmfit.model import ModelResult
import numpy as np
from typing import Optional

_nodes = []
try:
    import funcnodes_plotly as fn_plotly  # noqa: F401

    import plotly.graph_objects as go

    @fn.NodeDecorator(
        "lmfit.plot_results_plotly",
        description="Plots the results of a lmfit fit",
        outputs=[{"type": go.Figure, "name": "figure"}],
        default_render_options={
            "data": {
                "src": "figure",
            }
        },
    )
    def plot_results_plotly(
        fit_results: ModelResult,
        x: Optional[np.ndarray] = None,
        fig: Optional[go.Figure] = None,
    ) -> go.Figure:
        if x is None:
            x = fit_results.userkws["x"]

        comp_y = fit_results.eval_components(
            x=x,
        )

        if fig is None:
            fig = go.Figure()

        if fit_results.data is not None:
            fig.add_trace(
                go.Scatter(
                    x=fit_results.userkws["x"],
                    y=fit_results.data,
                    mode="markers" if len(fit_results.data) < 100 else "lines",
                    name="data",
                )
            )

        for name, data in comp_y.items():
            fig.add_trace(go.Scatter(x=x, y=data, mode="lines", name=name))

        fig.add_trace(go.Scatter(x=x, y=fit_results.best_fit, mode="lines", name="fit"))
        if fit_results.residual is not None:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=-fit_results.residual,
                    mode="lines",
                    name="residual",
                )
            )

        return fig

    _nodes.append(plot_results_plotly)

except ImportError:
    pass


PLOT_SHELF = fn.Shelf(
    nodes=_nodes,
    name="Plotting",
    description="Nodes for plotting lmfit results",
    subshelves=[],
)
