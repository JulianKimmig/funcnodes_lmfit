"""
here models are made
"""

import funcnodes as fn
from typing import Optional, Union, Literal, Tuple
from lmfit.model import CompositeModel, Model, ModelResult
from lmfit.models import (
    SplineModel,
    ExpressionModel,
    PolynomialModel,
    ThermalDistributionModel,
    StepModel,
    RectangleModel,
    ConstantModel,
    ComplexConstantModel,
    LinearModel,
    QuadraticModel,
    GaussianModel,
    LorentzianModel,
    SplitLorentzianModel,
    VoigtModel,
    PseudoVoigtModel,
    MoffatModel,
    Pearson4Model,
    Pearson7Model,
    StudentsTModel,
    BreitWignerModel,
    LognormalModel,
    DampedOscillatorModel,
    DampedHarmonicOscillatorModel,
    ExponentialGaussianModel,
    SkewedGaussianModel,
    SkewedVoigtModel,
    DoniachModel,
    PowerLawModel,
    ExponentialModel,
)
import numpy as np
from .utils import autoprefix, update_model_params, model_composit_train_create
from ._auto_model import (
    ConstantModel_node,
    ComplexConstantModel_node,
    LinearModel_node,
    QuadraticModel_node,
    GaussianModel_node,
    Gaussian2dModel_node,
    LorentzianModel_node,
    SplitLorentzianModel_node,
    VoigtModel_node,
    PseudoVoigtModel_node,
    MoffatModel_node,
    Pearson4Model_node,
    Pearson7Model_node,
    StudentsTModel_node,
    BreitWignerModel_node,
    LognormalModel_node,
    DampedOscillatorModel_node,
    DampedHarmonicOscillatorModel_node,
    ExponentialGaussianModel_node,
    SkewedGaussianModel_node,
    SkewedVoigtModel_node,
    DoniachModel_node,
    PowerLawModel_node,
    ExponentialModel_node,
)


# SplineModel,ExpressionModel,PolynomialModel,ThermalDistributionModel,StepModel,RectangleModel


@fn.NodeDecorator(
    "lmfit.SplineModel",
    description="Create or add a SplineModel model",
    outputs=[{"type": SplineModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "xknots_min": {"hidden": True},
        "xknots_max": {"hidden": True},
    },
)
def SplineModel_node(
    xknots: np.ndarray,
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    xknots_min: Optional[np.ndarray] = None,
    xknots_max: Optional[np.ndarray] = None,
) -> Model:
    xknots = np.asarray(xknots)
    xknots = xknots.flatten()
    if xknots_min is None:
        xknots_min = np.zeros_like(xknots) - np.inf
    if xknots_max is None:
        xknots_max = np.zeros_like(xknots) + np.inf

    if not xknots_min.shape == xknots.shape == xknots_max.shape:
        raise ValueError("xknots, xknots_min and xknots_max must have the same shape")

    prefix = autoprefix(SplineModel, composite)
    model = SplineModel(prefix=prefix, xknots=xknots)

    for i, xn in enumerate(xknots):
        name = f"{prefix}s{i}"
        model.set_param_hint(
            name,
            min=xknots_min[i],
            max=xknots_max[i],
        )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    "lmfit.ExpressionModel",
    description="Create or add a ExpressionModel model",
    outputs=[{"type": Model, "name": "model"}],
)
def ExpressionModel_node(
    expression: str,
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    independent_vars: Optional[Union[str, list[str]]] = None,
) -> Model:
    if independent_vars is not None:
        if isinstance(independent_vars, str):
            independent_vars = independent_vars.split(",")
        independent_vars = [s.strip() for s in independent_vars]
    # expresssion model has no prefix, the vars are as in the expression
    # prefix = autoprefix(ExpressionModel, composite)

    model = ExpressionModel(expression, independent_vars=independent_vars)

    # set all params to 1
    for param in model.param_names:
        model.set_param_hint(param, value=1)

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    "lmfit.PolynomialModel",
    description="Create or add a PolynomialModel model",
    outputs=[{"type": Model, "name": "model"}],
    default_io_options={
        "degree": {
            "value_options": {"max": 7, "min": 0},
        },
        "model": {"on": {"after_set_value": update_model_params()}},
        "min_c": {"hidden": True},
        "max_c": {"hidden": True},
    },
)
def PolynomialModel_node(
    degree: int,
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    min_c: float = -(10**8),
    max_c: float = 10**8,
) -> Model:
    prefix = autoprefix(PolynomialModel, composite)

    # clip degree to 7
    degree = int(min(7, max(0, degree)))
    model = PolynomialModel(degree, prefix=prefix)

    # default bounds is inf which does not work for polynomial models
    for i in range(degree + 1):
        model.set_param_hint(f"{prefix}c{i}", min=min_c, max=max_c)

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    "lmfit.ThermalDistributionModel",
    description="Create or add a ThermalDistributionModel model",
    outputs=[{"type": Model, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "amplitude_value": {"hidden": True},
        "amplitude_min": {"hidden": True},
        "amplitude_max": {"hidden": True},
        "amplitude_vary": {"hidden": True},
        "center_value": {"hidden": True},
        "center_min": {"hidden": True},
        "center_max": {"hidden": True},
        "center_vary": {"hidden": True},
        "kt_value": {"hidden": True},
        "kt_min": {"hidden": True},
        "kt_max": {"hidden": True},
        "kt_vary": {"hidden": True},
    },
)
def ThermalDistributionModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    form: Literal["bose", "maxwell", "fermi"] = "bose",
    amplitude_value: Optional[float] = 1,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: Optional[bool] = True,
    center_value: Optional[float] = 0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: Optional[bool] = True,
    kt_value: Optional[float] = 1,
    kt_min: Optional[float] = None,
    kt_max: Optional[float] = None,
    kt_vary: Optional[bool] = True,
) -> Model:
    prefix = autoprefix(ThermalDistributionModel, composite)
    model = ThermalDistributionModel(prefix=prefix, form=form)
    model.set_param_hint(
        "amplitude",
        value=amplitude_value,
        min=amplitude_min if amplitude_min is not None else -np.inf,
        max=amplitude_max if amplitude_max is not None else np.inf,
        vary=amplitude_vary,
    )

    model.set_param_hint(
        "center",
        value=center_value,
        min=center_min if center_min is not None else -np.inf,
        max=center_max if center_max is not None else np.inf,
        vary=center_vary,
    )

    model.set_param_hint(
        "kt",
        value=kt_value,
        min=kt_min if kt_min is not None else -np.inf,
        max=kt_max if kt_max is not None else np.inf,
        vary=kt_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    "lmfit.StepModel",
    description="Create or add a StepModel model",
    outputs=[{"type": Model, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "amplitude_value": {"hidden": True},
        "amplitude_min": {"hidden": True},
        "amplitude_max": {"hidden": True},
        "amplitude_vary": {"hidden": True},
        "center_value": {"hidden": True},
        "center_min": {"hidden": True},
        "center_max": {"hidden": True},
        "center_vary": {"hidden": True},
        "sigma_value": {"hidden": True},
        "sigma_min": {"hidden": True},
        "sigma_max": {"hidden": True},
        "sigma_vary": {"hidden": True},
    },
)
def StepModel_node(
    composite: Optional[CompositeModel] = None,
    y: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    form: Literal["linear", "atan", "erf", "logistic"] = "linear",
    amplitude_value: Optional[float] = 1,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: Optional[bool] = True,
    center_value: Optional[float] = 0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: Optional[bool] = True,
    sigma_value: Optional[float] = 1,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    sigma_vary: Optional[bool] = True,
) -> Model:
    prefix = autoprefix(StepModel, composite)
    model = StepModel(prefix=prefix, form=form)
    model.set_param_hint(
        "amplitude",
        value=amplitude_value,
        min=amplitude_min if amplitude_min is not None else -np.inf,
        max=amplitude_max if amplitude_max is not None else np.inf,
        vary=amplitude_vary,
    )

    model.set_param_hint(
        "center",
        value=center_value,
        min=center_min if center_min is not None else -np.inf,
        max=center_max if center_max is not None else np.inf,
        vary=center_vary,
    )

    model.set_param_hint(
        "sigma",
        value=sigma_value,
        min=sigma_min if sigma_min is not None else -np.inf,
        max=sigma_max if sigma_max is not None else np.inf,
        vary=sigma_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    "lmfit.RectangleModel",
    description="Create or add a RectangleModel model",
    outputs=[{"type": Model, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "amplitude_value": {"hidden": True},
        "amplitude_min": {"hidden": True},
        "amplitude_max": {"hidden": True},
        "amplitude_vary": {"hidden": True},
        "center1_value": {"hidden": True},
        "center1_min": {"hidden": True},
        "center1_max": {"hidden": True},
        "center1_vary": {"hidden": True},
        "center2_value": {"hidden": True},
        "center2_min": {"hidden": True},
        "center2_max": {"hidden": True},
        "center2_vary": {"hidden": True},
        "sigma1_value": {"hidden": True},
        "sigma1_min": {"hidden": True},
        "sigma1_max": {"hidden": True},
        "sigma1_vary": {"hidden": True},
        "sigma2_value": {"hidden": True},
        "sigma2_min": {"hidden": True},
        "sigma2_max": {"hidden": True},
        "sigma2_vary": {"hidden": True},
    },
)
def RectangleModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    form: Literal["linear", "atan", "erf", "logistic"] = "linear",
    amplitude_value: Optional[float] = 1,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: Optional[bool] = True,
    center1_value: Optional[float] = 0,
    center1_min: Optional[float] = None,
    center1_max: Optional[float] = None,
    center1_vary: Optional[bool] = True,
    center2_value: Optional[float] = 1,
    center2_min: Optional[float] = None,
    center2_max: Optional[float] = None,
    center2_vary: Optional[bool] = True,
    sigma1_value: Optional[float] = 1,
    sigma1_min: Optional[float] = None,
    sigma1_max: Optional[float] = None,
    sigma1_vary: Optional[bool] = True,
    sigma2_value: Optional[float] = 1,
    sigma2_min: Optional[float] = None,
    sigma2_max: Optional[float] = None,
    sigma2_vary: Optional[bool] = True,
) -> Model:
    prefix = autoprefix(RectangleModel, composite)
    model = RectangleModel(prefix=prefix, form=form)
    model.set_param_hint(
        "amplitude",
        value=amplitude_value,
        min=amplitude_min if amplitude_min is not None else -np.inf,
        max=amplitude_max if amplitude_max is not None else np.inf,
        vary=amplitude_vary,
    )

    model.set_param_hint(
        "center1",
        value=center1_value,
        min=center1_min if center1_min is not None else -np.inf,
        max=center1_max if center1_max is not None else np.inf,
        vary=center1_vary,
    )

    model.set_param_hint(
        "center2",
        value=center2_value,
        min=center2_min if center2_min is not None else -np.inf,
        max=center2_max if center2_max is not None else np.inf,
        vary=center2_vary,
    )

    model.set_param_hint(
        "sigma1",
        value=sigma1_value,
        min=sigma1_min if sigma1_min is not None else -np.inf,
        max=sigma1_max if sigma1_max is not None else np.inf,
        vary=sigma1_vary,
    )

    model.set_param_hint(
        "sigma2",
        value=sigma2_value,
        min=sigma2_min if sigma2_min is not None else -np.inf,
        max=sigma2_max if sigma2_max is not None else np.inf,
        vary=sigma2_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    "lmfit.model_merge",
    description="Merges two lmfit models with a defined operator",
    outputs=[{"name": "model"}],
)
def merge_models(
    a: Model, b: Model, operator: Literal["+", "-", "*", "/"] = "+"
) -> CompositeModel:
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    if operator == "/":
        return a / b
    raise ValueError(f"unknown operator: {operator}")


AUTOMODELS = [
    ConstantModel,
    ComplexConstantModel,
    LinearModel,
    QuadraticModel,
    GaussianModel,
    LorentzianModel,
    SplitLorentzianModel,
    VoigtModel,
    PseudoVoigtModel,
    MoffatModel,
    Pearson4Model,
    Pearson7Model,
    StudentsTModel,
    BreitWignerModel,
    LognormalModel,
    DampedOscillatorModel,
    DampedHarmonicOscillatorModel,
    ExponentialGaussianModel,
    SkewedGaussianModel,
    SkewedVoigtModel,
    DoniachModel,
    PowerLawModel,
    ExponentialModel,
]

AUTOMODELMAP = {model.__name__: model for model in AUTOMODELS}


@fn.NodeDecorator(
    "lmfit.auto_model",
    description="Automatically generate a model from the data",
    outputs=[{"type": Model, "name": "model"}, {"type": ModelResult, "name": "result"}],
    default_io_options={
        "r2_threshold": {
            "value": 0.95,
            "value_options": {"min": 0, "max": 1},
        },
    },
    seperate_thread=True,
)
def auto_model(
    x: np.ndarray, y: np.ndarray, r2_threshold: float = 0.95, iterations: int = 1
) -> Tuple[Model, ModelResult]:
    """
    Automatically generate a model from the data

    Parameters
    ----------
    x : np.ndarray
        x data
    y : np.ndarray
        y data
    r2_threshold : float, optional
        R^2 threshold for the model, by default 0.95
        Models will be generated until the R^2 is above this threshold or no more models can be generated.

    Returns
    -------
    Tuple[Model, ModelResult]
        The model and the fit result
    """

    r2_threshold = max(0, min(1, r2_threshold))

    composite_model = None
    composite_model_fit = None
    iterations = int(max(1, iterations))
    for i in range(iterations):
        best_model = None
        best_r2 = -np.inf

        if composite_model_fit is not None:
            y_res = y - composite_model_fit.eval(x=x)
        else:
            y_res = y

        for modelclass in AUTOMODELS:
            prefix = autoprefix(modelclass, composite_model)
            model: Model = modelclass(prefix=prefix)
            try:
                guess = model.guess(data=y_res, x=x)
            except Exception:
                guess = model.make_params()
            try:
                fit: ModelResult = model.fit(data=y_res, params=guess, x=x)
            except Exception:
                continue
            r2 = fit.rsquared
            if r2 > best_r2:
                # update model parameters to the best fit
                for param in fit.params:
                    model.set_param_hint(param, value=fit.params[param].value)

                best_r2 = r2
                best_model = model

            if best_r2 >= r2_threshold:
                break

        if composite_model is None:
            composite_model = best_model
        else:
            composite_model += best_model

        # final fit
        composite_model_fit = composite_model.fit(data=y, x=x)

        for param in composite_model_fit.params:
            composite_model.set_param_hint(
                param, value=composite_model_fit.params[param].value
            )

    return composite_model, composite_model_fit


@fn.NodeDecorator(
    "lmfit.quickmodel",
    description="Quick model creation",
    outputs=[{"type": Model, "name": "model"}],
    default_io_options={
        "modelname": {"value_options": {"options": list(AUTOMODELMAP.keys())}},
    },
)
def quickmodel(
    modelname: str,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    composite: Optional[CompositeModel] = None,
):
    """
    Quick model creation
    """
    modelclass = AUTOMODELMAP.get(modelname)
    if modelclass is None:
        raise ValueError(f"Unknown model: {modelname}")

    prefix = autoprefix(modelclass, composite)
    model = modelclass(prefix=prefix)

    return model_composit_train_create(model, composite, x, y)


MODEL_SHELF = fn.Shelf(
    name="Models",
    description="Lmfit models",
    nodes=[
        auto_model,
        quickmodel,
        ConstantModel_node,
        ComplexConstantModel_node,
        LinearModel_node,
        QuadraticModel_node,
        GaussianModel_node,
        Gaussian2dModel_node,
        LorentzianModel_node,
        SplitLorentzianModel_node,
        VoigtModel_node,
        PseudoVoigtModel_node,
        MoffatModel_node,
        Pearson4Model_node,
        Pearson7Model_node,
        StudentsTModel_node,
        BreitWignerModel_node,
        LognormalModel_node,
        DampedOscillatorModel_node,
        DampedHarmonicOscillatorModel_node,
        ExponentialGaussianModel_node,
        SkewedGaussianModel_node,
        SkewedVoigtModel_node,
        DoniachModel_node,
        PowerLawModel_node,
        ExponentialModel_node,
        SplineModel_node,
        ExpressionModel_node,
        PolynomialModel_node,
        ThermalDistributionModel_node,
        StepModel_node,
        RectangleModel_node,
    ],
    subshelves=[
        fn.Shelf(
            name="Model Operations",
            description="Operations with lmfit Models",
            nodes=[merge_models],
            subshelves=[],
        )
    ],
)
