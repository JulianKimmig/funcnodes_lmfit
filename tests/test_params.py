from funcnodes_lmfit.params import update_model_param, guess_model
from unittest import IsolatedAsyncioTestCase
from lmfit.models import GaussianModel, Model
import funcnodes as fn
import numpy as np

fn.config.IN_NODE_TEST = True


class TestParamNodes(IsolatedAsyncioTestCase):
    async def test_update_model_param(self):
        model = GaussianModel()
        ins = update_model_param()
        ins.inputs["model"].value = model
        ins.inputs["param_name"].value = "amplitude"
        ins.inputs["value"].value = 2.0
        await ins
        model: Model = ins.outputs["updated_model"].value

        self.assertEqual(model.make_params()["amplitude"].value, 2.0)

    async def test_guess_model(self):
        model = GaussianModel()
        x = np.linspace(0, 10, 1000)
        center = 5
        amplitude = 3
        sigma = 0.5

        y = (amplitude / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -((x - center) ** 2) / (2 * sigma**2)
        )
        ins = guess_model()
        ins.inputs["model"].value = model
        ins.inputs["x"].value = x
        ins.inputs["y"].value = y
        await ins
        model: Model = ins.outputs["updated_model"].value

        self.assertIsInstance(model, Model)

        parameters = ins.outputs["parameters"].value
        self.assertIsInstance(parameters, dict)

        self.assertEqual(parameters["center"].value, center)
        self.assertAlmostEqual(parameters["sigma"].value, sigma, delta=0.1)
        self.assertAlmostEqual(parameters["amplitude"].value, amplitude, delta=1.5)

        self.assertEqual(model.make_params()["center"].value, center)
        self.assertAlmostEqual(
            model.make_params()["amplitude"].value, amplitude, delta=1.5
        )
        self.assertAlmostEqual(model.make_params()["sigma"].value, sigma, delta=0.1)
