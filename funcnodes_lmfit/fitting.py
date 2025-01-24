from typing import Optional
from lmfit import Model
import numpy as np
from lmfit.model import ModelResult
import funcnodes as fn
from tqdm import tqdm


@fn.NodeDecorator(
    "lmfit.fit",
    description="Fits a lmfit Model to data",
    outputs=[{"type": ModelResult, "name": "result"}],
)
def fit(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    try_guess: bool = True,
    node: Optional[fn.Node] = None,
) -> ModelResult:
    x = np.asarray(x)
    y = np.asarray(y)
    params = model.make_params()
    if try_guess:
        try:
            params = model.guess(y, x=x)
        except Exception:
            pass

    _tqdm_kwargs = {
        "desc": "Fitting composite",
    }
    if node is not None:
        progress = node.progress(**_tqdm_kwargs)
    else:
        progress = tqdm(**_tqdm_kwargs)

    def _cb(params, iter, resid, *args, **kws):
        progress.update(1)

    fit_results = model.fit(y, params=params, x=x, iter_cb=_cb)
    progress.close()

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
