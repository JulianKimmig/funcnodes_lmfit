from lmfit import Model
import numpy as np
from lmfit.model import ModelResult
import funcnodes as fn


@fn.NodeDecorator(
    "lmfit.fit",
    description="Fits a lmfit Model to data",
    outputs=[{"type": ModelResult, "name": "result"}],
)
def fit(
    x: np.ndarray, y: np.ndarray, model: Model, try_guess: bool = True
) -> ModelResult:
    x = np.asarray(x)
    y = np.asarray(y)
    params = model.make_params()
    if try_guess:
        try:
            params = model.guess(y, x=x)
        except Exception:
            pass

    fit_results = model.fit(y, params=params, x=x)

    return fit_results


@fn.NodeDecorator(
    "lmfit.fit_summary",
    description="Summarizes the fit results",
    outputs=[{"name": "summary"}],
)
def fit_summary(fit_results: ModelResult) -> dict:
    return fit_results.summary()


@fn.NodeDecorator(
    "lmfit.fit_report",
    description="Generates a report of the fit results",
    outputs=[{"name": "report"}],
)
def fit_report(fit_results: ModelResult) -> str:
    return fit_results.fit_report()


FIT_SHELF = fn.Shelf(
    nodes=[fit, fit_summary, fit_report],
    name="Fitting",
    description="Nodes for fitting lmfit models",
    subshelves=[],
)
