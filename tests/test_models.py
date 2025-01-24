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
    itermodel,
    reduce_composite_node,
)
from funcnodes_lmfit.use_model import predict, PANDAS_INSTALLED

if PANDAS_INSTALLED:
    import pandas as pd
    from funcnodes_lmfit.use_model import predict_df

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

    async def test_itermodel(self):
        x = np.linspace(0, 10, 1000)
        ITERS = 6
        rng = np.random.default_rng(42)
        centers = np.linspace(2, 8, ITERS) + rng.normal(0, 0.1, ITERS)
        sigmas = 0.2 + rng.random(ITERS) * 0.4
        amplitudes = 0.5 + rng.random(ITERS)
        model = None
        for i in range(ITERS):
            _model = GaussianModel(prefix=f"gaussian{i}")
            _model.set_param_hint("center", value=centers[i])
            _model.set_param_hint("sigma", value=sigmas[i])
            _model.set_param_hint("amplitude", value=amplitudes[i])

            if model is None:
                model = _model
            else:
                model += _model

        params = model.make_params()
        y = model.eval(params, x=x)

        node = itermodel()
        basemodel = GaussianModel()

        node.inputs["basemodel"].value = basemodel
        node.inputs["r2_threshold"].value = 0.998
        node.inputs["x"].value = x
        node.inputs["y"].value = y
        node.inputs["max_iterations"].value = ITERS + 1

        await node

        model: Model = node.outputs["model"].value
        res = node.outputs["result"].value

        self.assertIsInstance(res, ModelResult)
        self.assertGreater(res.rsquared, 0.95)

        self.assertIsInstance(model, CompositeModel)
        self.assertEqual(len(model.components), ITERS)

    async def test_reduce_composite(self):
        x = np.linspace(0, 10, 1000)
        ITERS = 6
        rng = np.random.default_rng(42)
        centers = np.linspace(2, 8, ITERS) + rng.normal(0, 0.1, ITERS)
        sigmas = 0.2 + rng.random(ITERS) * 0.4
        amplitudes = 0.5 + rng.random(ITERS)
        model = None
        for i in range(ITERS * 2):
            _model = GaussianModel(prefix=f"gaussian{i}")
            _model.set_param_hint("center", value=centers[i % ITERS])
            _model.set_param_hint("sigma", value=sigmas[i % ITERS])
            _model.set_param_hint("amplitude", value=amplitudes[i % ITERS])

            if model is None:
                model = _model
            else:
                model += _model

        y = model.eval(model.make_params(), x=x)

        node = reduce_composite_node()
        print(node.outputs)
        node.inputs["model"].value = model
        node.inputs["x"].value = x
        node.inputs["y"].value = y
        node.inputs["r2_threshold"].value = 0.998

        await node

        model: Model = node.outputs["red_model"].value
        res = node.outputs["result"].value

        self.assertIsInstance(res, ModelResult)
        self.assertGreater(res.rsquared, 0.95)

        self.assertIsInstance(model, CompositeModel)

        self.assertEqual(len(model.components), ITERS)


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


class TestUseModel(IsolatedAsyncioTestCase):
    async def test_predict(self):
        x = np.linspace(0, 10, 100)
        model = GaussianModel()
        params = model.make_params()
        y = model.eval(params, x=x)

        out = predict()
        out.inputs["x"].value = x
        out.inputs["model"].value = model

        await out

        prediction, components = (
            out.outputs["prediction"].value,
            out.outputs["components"].value,
        )

        self.assertIsInstance(prediction, np.ndarray)
        self.assertIsInstance(components, np.ndarray)

        self.assertEqual(prediction.shape, (100,))
        self.assertEqual(components.shape, (100, 1))
        self.assertTrue(np.allclose(prediction, y))

    async def test_predict_composite(self):
        x = np.linspace(0, 10, 100)
        model = GaussianModel(prefix="gaussian") + LinearModel(prefix="linear")

        params = model.make_params()
        y = model.eval(params, x=x)

        out = predict()
        out.inputs["x"].value = x
        out.inputs["model"].value = model

        await out

        prediction, components = (
            out.outputs["prediction"].value,
            out.outputs["components"].value,
        )

        self.assertIsInstance(prediction, np.ndarray)
        self.assertIsInstance(components, np.ndarray)

        self.assertEqual(prediction.shape, (100,))
        self.assertEqual(components.shape, (100, 2))

        self.assertTrue(np.allclose(prediction, y))

        self.assertTrue(
            np.allclose(components[:, 0], model.eval_components(x=x)["gaussian"])
        )
        self.assertTrue(
            np.allclose(components[:, 1], model.eval_components(x=x)["linear"])
        )

    if PANDAS_INSTALLED:

        async def test_predict_df(self):
            x = np.linspace(0, 10, 100)
            model = GaussianModel()
            params = model.make_params()
            y = model.eval(params, x=x)

            out = predict_df()
            out.inputs["x"].value = x
            out.inputs["model"].value = model

            await out

            prediction = out.outputs["predictions"].value

            self.assertIsInstance(prediction, pd.DataFrame)

            self.assertEqual(prediction.shape, (100, 1))
            self.assertEqual(prediction.columns, [model._name])

            self.assertTrue(np.allclose(prediction[model._name], y))

        async def test_predict_df_composite(self):
            x = np.linspace(0, 10, 100)
            model = GaussianModel(prefix="gaussian") + LinearModel(prefix="linear")

            params = model.make_params()
            y = model.eval(params, x=x)

            out = predict_df()
            out.inputs["x"].value = x
            out.inputs["model"].value = model

            await out

            prediction = out.outputs["predictions"].value

            self.assertIsInstance(prediction, pd.DataFrame)

            self.assertEqual(prediction.shape, (100, 3))
            self.assertEqual(
                prediction.columns.tolist(),
                [
                    "(Model(gaussian, prefix='gaussian') + Model(linear, prefix='linear'))",
                    "gaussian",
                    "linear",
                ],
            )

            self.assertTrue(
                np.allclose(
                    prediction[
                        "(Model(gaussian, prefix='gaussian') + Model(linear, prefix='linear'))"
                    ],
                    y,
                )
            )
            self.assertTrue(
                np.allclose(
                    prediction["gaussian"], model.eval_components(x=x)["gaussian"]
                )
            )

            self.assertTrue(
                np.allclose(prediction["linear"], model.eval_components(x=x)["linear"])
            )
