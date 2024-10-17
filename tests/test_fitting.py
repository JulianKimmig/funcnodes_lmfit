from unittest import IsolatedAsyncioTestCase
from funcnodes_lmfit.fitting import fit, fit_summary, fit_report
from funcnodes_lmfit.model import LinearModel_node, ExpressionModel_node
import numpy as np
import funcnodes as fn
from lmfit.model import CompositeModel

fn.config.IN_NODE_TEST = True


class TestFitting(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.x = np.linspace(0, 10, 100)
        self.y = 2 * self.x + 1 + 1.1 / (self.x + 2)

        self.linmodel = LinearModel_node()
        self.expmodel = ExpressionModel_node()
        self.expmodel.inputs["expression"].value = "a/(x+b)"

        self.linmodel.outputs["model"].connect(self.expmodel.inputs["composite"])
        await self.linmodel
        await self.expmodel
        self.assertIsInstance(self.expmodel.outputs["model"].value, CompositeModel)

    async def test_fit(self):
        fitins = fit()
        fitins.inputs["x"].value = self.x
        fitins.inputs["y"].value = self.y
        fitins.inputs["model"].connect(self.expmodel.outputs["model"])

        await fitins

        self.assertTrue(fitins.outputs["result"].value.success)

    async def test_fit_summary(self):
        fitins = fit()
        fitins.inputs["x"].value = self.x
        fitins.inputs["y"].value = self.y
        fitins.inputs["model"].connect(self.expmodel.outputs["model"])

        await fitins

        summary = fit_summary()
        summary.inputs["fit_results"].connect(fitins.outputs["result"])

        await summary

        _sum = summary.outputs["summary"].value
        self.assertIsInstance(_sum, dict)
        self.assertIn("best_values", _sum, _sum)
        best_values = _sum["best_values"]
        self.assertIsInstance(best_values, dict)

    async def test_fit_report(self):
        fitins = fit()
        fitins.inputs["x"].value = self.x
        fitins.inputs["y"].value = self.y
        fitins.inputs["model"].connect(self.expmodel.outputs["model"])

        await fitins

        report = fit_report()
        report.inputs["fit_results"].connect(fitins.outputs["result"])

        await report

        _report = report.outputs["report"].value

        self.assertIsInstance(_report, str)
