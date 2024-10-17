from unittest import IsolatedAsyncioTestCase
from funcnodes_lmfit.fitting import fit
from funcnodes_lmfit.model import LinearModel_node, ExpressionModel_node
import numpy as np
import funcnodes as fn

fn.config.IN_NODE_TEST = True


class TestPlotFit(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.x = np.linspace(0, 10, 100)
        self.y = 2 * self.x + 1 + 1.1 / (self.x + 2)

        self.linmodel = LinearModel_node()
        self.exprmodel = ExpressionModel_node()
        self.exprmodel.inputs["expression"].value = "a/(x+b)"

        self.linmodel.outputs["model"].connect(self.exprmodel.inputs["composite"])

        self.fitins = fit()
        self.fitins.inputs["x"].value = self.x
        self.fitins.inputs["y"].value = self.y
        self.fitins.inputs["model"].connect(self.exprmodel.outputs["model"])

        await self.linmodel
        await self.exprmodel
        await self.fitins

    async def test_plotly(self):
        try:
            from funcnodes_lmfit.plotting import plot_results_plotly
            from plotly import graph_objects as go
        except ImportError:
            return

        plotly = plot_results_plotly()
        plotly.inputs["fit_results"].connect(self.fitins.outputs["result"])
        plotly.inputs["x"].value = self.x
        await plotly

        fig = plotly.outputs["figure"].value
        self.assertIsInstance(fig, go.Figure)
