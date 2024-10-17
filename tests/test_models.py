from unittest import IsolatedAsyncioTestCase
import funcnodes as fn
from lmfit.model import Model, CompositeModel, ModelResult
from lmfit.models import GaussianModel, LinearModel
import numpy as np
from funcnodes_lmfit._auto_model import _AUTOMODELS

from funcnodes_lmfit.model import (
    SplineModel_node,
    ExpressionModel_node,
    PolynomialModel_node,
    ThermalDistributionModel_node,
    StepModel_node,
    RectangleModel_node,
    merge_models,
    auto_model,
    quickmodel,
)

fn.config.IN_NODE_TEST = True


class TestModels(IsolatedAsyncioTestCase):
    """ """

    async def test_auto_models(self):
        x0_test = np.linspace(0, 10, 100)
        x1_test = np.linspace(0, 10, 100)

        for node in _AUTOMODELS:
            print(node)
            instance = node()
            await instance

            model: Model = instance.outputs["model"].value
            self.assertIsInstance(model, Model)

            params = model.make_params()

            indipendent_input = {}
            for v, x in zip(model.independent_vars, [x0_test, x1_test]):
                indipendent_input[v] = x

            ini_params = params.copy()

            y_test = model.eval(params, **indipendent_input)
            self.assertIsInstance(y_test, np.ndarray)
            self.assertEqual(y_test.shape, (100,))

            # randomly mutate the parameters
            for param in params.values():
                # mutate the parameter value by 10%
                param.value *= np.random.uniform(0.9, 1.1)
                # add some noise to the parameter value
                param.value += np.random.normal(0, 0.01)

            # check if the parameters have changed
            for paramname, param in params.items():
                self.assertNotEqual(param.value, ini_params[paramname].value)

            y_ini = model.eval(params, **indipendent_input)

            # compare y_ini and y_test
            self.assertFalse(np.allclose(y_ini, y_test))

            # fit the model to the test data
            fit_result = model.fit(data=y_test, params=params, **indipendent_input)

            # check if the fit was successful
            self.assertTrue(fit_result.success)

            y_fitted = model.eval(fit_result.params, **indipendent_input)

            # compare y_fitted and y_test
            self.assertTrue(np.allclose(y_fitted, y_test), (node, y_fitted, y_test))

    async def test_SplineModel_node(self):
        x_test = np.linspace(0, 10, 100)
        inst = SplineModel_node()
        await inst
        # no knots given, no model should be created
        self.assertEqual(inst.outputs["model"].value, fn.NoValue)

        inst.inputs["xknots"].value = np.linspace(0, 10, 5)
        await inst

        model: Model = inst.outputs["model"].value
        self.assertIsInstance(model, Model)

        params = model.make_params()
        ini_params = params.copy()

        self.assertEqual(len(params), 5, params)

        y_test = model.eval(params, x=x_test)

        print(y_test)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertEqual(y_test.shape, (100,))

        # randomly mutate the parameters
        for param in params.values():
            # mutate the parameter value by 10%
            param.value *= np.random.uniform(0.9, 1.1)
            # add some noise to the parameter value
            param.value += np.random.normal(0, 0.1)

        for paramname, param in params.items():
            print(paramname, param.value, ini_params[paramname].value)
        # check if the parameters have changed
        for paramname, param in params.items():
            print(paramname, param.value, ini_params[paramname].value)
            self.assertNotEqual(param.value, ini_params[paramname].value)

        y_ini = model.eval(params, x=x_test)

        # compare y_ini and y_test
        self.assertFalse(np.allclose(y_ini, y_test))

        # fit the model to the test data

        fit_result = model.fit(data=y_test, params=params, x=x_test)

        # check if the fit was successful
        self.assertTrue(fit_result.success)

        y_fitted = model.eval(fit_result.params, x=x_test)

        # compare y_fitted and y_test
        self.assertTrue(np.allclose(y_fitted, y_test))

    async def test_ExpressionModel_node(self):
        x_test = np.linspace(0, 10, 100)
        inst = ExpressionModel_node()
        await inst
        # no func
        self.assertEqual(inst.outputs["model"].value, fn.NoValue)

        inst.inputs["expression"].value = "a*x**2 + b*x + c"
        y_target = 2 * x_test**2 + 3 * x_test + 4
        await inst

        model: Model = inst.outputs["model"].value
        self.assertIsInstance(model, Model)

        params = model.make_params()

        self.assertEqual(len(params), 3, params)

        y_ini = model.eval(params, x=x_test)

        # compare y_ini and y_test
        self.assertFalse(np.allclose(y_ini, y_target))

        # fit the model to the test data

        fit_result = model.fit(data=y_target, params=params, x=x_test)

        # check if the fit was successful
        self.assertTrue(fit_result.success)

        y_fitted = model.eval(fit_result.params, x=x_test)

        # compare y_fitted and y_test
        self.assertTrue(np.allclose(y_fitted, y_target))

    async def test_PolynomialModel_node(self):
        x_test = np.linspace(0, 10, 100)
        inst = PolynomialModel_node()
        await inst
        # no degree
        self.assertEqual(inst.outputs["model"].value, fn.NoValue)

        inst.inputs["degree"].value = 2
        inst.inputs["min_c"].value = -10
        inst.inputs["max_c"].value = 10
        y_target = 2 * x_test**2 + 3 * x_test + 4
        await inst

        model: Model = inst.outputs["model"].value
        self.assertIsInstance(model, Model)

        params = model.make_params()

        self.assertEqual(len(params), 3, params)

        y_ini = model.eval(params, x=x_test)

        # compare y_ini and y_test
        self.assertFalse(np.allclose(y_ini, y_target))

        # fit the model to the test data

        fit_result = model.fit(data=y_target, params=params, x=x_test)

        # check if the fit was successful
        self.assertTrue(fit_result.success)

        y_fitted = model.eval(fit_result.params, x=x_test)

        # compare y_fitted and y_test
        self.assertTrue(np.allclose(y_fitted, y_target), (y_fitted, y_target))

    async def test_ThermalDistributionModel_node(self):
        x_test = np.linspace(-4, 5, 100) * (273 * 8.617e-5) + 1
        inst = ThermalDistributionModel_node()
        await inst

        model: Model = inst.outputs["model"].value
        self.assertIsInstance(model, Model)

        params = model.make_params()
        self.assertEqual(len(params), 3, params)

        y_ini = model.eval(params, x=x_test)
        y_target = 1 / (2 * np.exp((x_test - 100) / 273) - 1)

        # compare y_ini and y_test
        self.assertFalse(np.allclose(y_ini, y_target))

        # fit the model to the test data

        fit_result = model.fit(data=y_target, params=params, x=x_test)

        # check if the fit was successful
        self.assertTrue(fit_result.success)

        y_fitted = model.eval(fit_result.params, x=x_test)

        # compare y_fitted and y_test
        self.assertTrue(
            np.allclose(y_fitted, y_target), (y_fitted, y_target, fit_result.params)
        )

    async def test_StepModel_node(self):
        x_test = np.linspace(-10, 10, 100)
        inst = StepModel_node()
        await inst

        model: Model = inst.outputs["model"].value
        self.assertIsInstance(model, Model)

        params = model.make_params()
        self.assertEqual(len(params), 3, params)

        y_ini = model.eval(params, x=x_test)
        y_target = np.heaviside(x_test - 2, 1) * 3

        # compare y_ini and y_test
        self.assertFalse(np.allclose(y_ini, y_target))

        # fit the model to the test data

        fit_result = model.fit(data=y_target, params=params, x=x_test)

        # check if the fit was successful
        self.assertTrue(fit_result.success)

        y_fitted = model.eval(fit_result.params, x=x_test)

        # compare y_fitted and y_test
        self.assertTrue(
            np.allclose(y_fitted, y_target), (y_fitted, y_target, fit_result.params)
        )

    async def test_RectangleModel_node(self):
        x_test = np.linspace(-10, 10, 100)
        inst = RectangleModel_node()
        await inst

        model: Model = inst.outputs["model"].value
        self.assertIsInstance(model, Model)

        params = model.make_params()
        self.assertEqual(
            len(params), 6, params
        )  # 2*center, 2*sigma, amplitude, midpoint(dependent)

        y_ini = model.eval(params, x=x_test)
        y_target = np.zeros_like(x_test)
        y_target[40:60] = 3

        # compare y_ini and y_test
        self.assertFalse(np.allclose(y_ini, y_target))

        # fit the model to the test data
        fit_result = model.fit(data=y_target, params=params, x=x_test)

        # check if the fit was successful
        self.assertTrue(fit_result.success)

        y_fitted = model.eval(fit_result.params, x=x_test)

        # compare y_fitted and y_test
        self.assertTrue(
            np.allclose(y_fitted, y_target), (y_fitted, y_target, fit_result.params)
        )

    async def test_auto_model(self):
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + 1.1 / (x + 2)
        ins = auto_model()
        ins.inputs["x"].value = x
        ins.inputs["y"].value = y
        ins.inputs["r2_threshold"].value = 0.95
        ins.inputs["iterations"].value = 1

        await ins

        model: Model = ins.outputs["model"].value
        result = ins.outputs["result"].value

        self.assertIsInstance(model, Model)
        self.assertIsInstance(result, ModelResult)

        self.assertTrue(result.success)

    async def test_quickmodel(self):
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + 1.1 / (x + 2)
        ins = quickmodel()
        ins.inputs["x"].value = x
        ins.inputs["y"].value = y
        ins.inputs["modelname"].value = "GaussianModel"

        await ins

        model: Model = ins.outputs["model"].value

        self.assertIsInstance(model, GaussianModel)


class TestModelOperations(IsolatedAsyncioTestCase):
    async def test_merge_add(self):
        a = GaussianModel()
        b = LinearModel()

        merge = merge_models()
        merge.inputs["a"].value = a
        merge.inputs["b"].value = b
        merge.inputs["operator"].value = "+"

        await merge

        out = merge.outputs["model"].value

        self.assertIsInstance(out, CompositeModel)

    async def test_merge_subs(self):
        a = GaussianModel()
        b = LinearModel()

        merge = merge_models()
        merge.inputs["a"].value = a
        merge.inputs["b"].value = b
        merge.inputs["operator"].value = "-"

        await merge

        out = merge.outputs["model"].value

        self.assertIsInstance(out, CompositeModel)

    async def test_merge_mul(self):
        a = GaussianModel()
        b = LinearModel()

        merge = merge_models()
        merge.inputs["a"].value = a
        merge.inputs["b"].value = b
        merge.inputs["operator"].value = "*"

        await merge

        out = merge.outputs["model"].value

        self.assertIsInstance(out, CompositeModel)

    async def test_merge_div(self):
        a = GaussianModel()
        b = LinearModel()

        merge = merge_models()
        merge.inputs["a"].value = a
        merge.inputs["b"].value = b
        merge.inputs["operator"].value = "/"

        await merge

        out = merge.outputs["model"].value

        self.assertIsInstance(out, CompositeModel)

    async def test_merge_unknown(self):
        a = GaussianModel()
        b = LinearModel()

        merge = merge_models()
        merge.inputs["a"].value = a
        merge.inputs["b"].value = b
        merge.inputs["operator"].value = "foo"

        with self.assertRaises(fn.NodeTriggerError):
            await merge
