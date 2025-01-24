from typing import Type, Union, Optional, Tuple
from lmfit.model import Model, CompositeModel, ModelResult
import funcnodes as fn
import numpy as np
from tqdm import tqdm
from copy import deepcopy


def model_base_name(model: Union[Model, Type[Model]]) -> str:
    if isinstance(model, CompositeModel) or issubclass(model, CompositeModel):
        raise ValueError("CompositeModel is not supported")

    try:
        return model.func.__name__
    except Exception:
        pass

    if isinstance(model, Model):
        return model.__class__.__name__
    else:
        return model.__name__


def autoprefix(model: Type[Model], composite: Optional[CompositeModel] = None) -> str:
    basename = model_base_name(model)
    if composite is None:
        return f"{basename}_{1}_"

    prefix = 1
    prefixes = [p.prefix for p in composite.components]
    while f"{basename}_{prefix}_" in prefixes:
        prefix += 1
    return f"{basename}_{prefix}_"


def update_model_params():
    def func(src: fn.NodeIO, result: Model):
        node = src.node
        if not node:
            return
        if isinstance(result, CompositeModel):
            result = result.components[-1]
        params = result.make_params()
        for paramname, paramrootname in zip(
            result.param_names, result._param_root_names
        ):
            value = params[paramname].value
            try:
                node.inputs[paramrootname + "_value"].set_value(
                    value, does_trigger=False
                )
            except KeyError:
                pass

    return func


def model_composit_train_create(
    model: Model,
    composite: Optional[CompositeModel] = None,
    x: Optional[Union[list, np.ndarray]] = None,
    y: Optional[Union[list, np.ndarray]] = None,
    node: Optional[fn.Node] = None,
) -> Model:
    if x is None and composite is not None:
        if hasattr(composite, "_current_x"):
            x = composite._current_x

    if y is None and composite is not None:
        if hasattr(composite, "_current_y"):
            y = composite._current_y

    if y is not None and x is not None:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[0] == y.shape[0]:
            if composite is not None:
                _tqdm_kwargs = {
                    "desc": "Fitting composite",
                }
                if node is not None:
                    progress = node.progress(**_tqdm_kwargs)
                else:
                    progress = tqdm(**_tqdm_kwargs)

                def _cb(params, iter, resid, *args, **kws):
                    progress.update(1)

                composite_model_fit = composite.fit(data=y, x=x, iter_cb=_cb)
                progress.close()
                y_res = y - composite_model_fit.eval(x=x)
            else:
                y_res = y
            try:
                params = model.guess(y_res, x=x)
            except Exception:
                params = model.make_params()

            try:
                _tqdm_kwargs = {
                    "desc": "Fitting model",
                }
                if node is not None:
                    progress = node.progress(**_tqdm_kwargs)
                else:
                    progress = tqdm(**_tqdm_kwargs)

                def _cb(params, iter, resid, *args, **kws):
                    progress.update(1)

                fit_res = model.fit(y_res, params=params, x=x, iter_cb=_cb)
                progress.close()
                for param in fit_res.params:
                    model.set_param_hint(param, value=fit_res.params[param].value)
            except Exception:
                pass

    if composite is not None:
        model = composite + model

    model._current_x = x
    model._current_y = y
    return model


def reduce_composite(
    model: CompositeModel,
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    r2_threshold: float = 0.95,
) -> Tuple[CompositeModel, Union[ModelResult, None]]:
    """
    Reduce a CompositeModel by removing components that contribute minimally to the overall fit.

    This function iteratively prunes components from a composite model if their individual contributions to the fit
    exceed a given R² threshold, ensuring that the reduced model still fits the data well. The pruning is done by
    recursively evaluating the left and right components of the composite, and removing those that independently
    satisfy the R² threshold.

    Parameters
    ----------
    model : CompositeModel
        The composite model to be reduced.
    y : np.ndarray
        The dependent variable (data to fit).
    x : Optional[np.ndarray], default=None
        The independent variable(s) corresponding to `y`. If `None`, the model will evaluate without `x`.
    r2_threshold : float, default=0.95
        The minimum R² value required for a sub-model to be considered sufficient to replace the composite
        in its subtree. Must be between 0 and 1.

    Returns
    -------
    CompositeModel
        The reduced composite model, where unnecessary components have been pruned.

    Notes
    -----
    - The function relies on the `rsquared` metric for assessing model sufficiency. This makes it suitable for cases
      where R² is a meaningful metric for fit quality.
    - A deep copy of the input model is made, so the original model remains unaltered.
    - The reduction process involves refitting sub-models and adjusting parameters as necessary.

    Example
    -------
    >>> from lmfit.models import GaussianModel
    >>> from lmfit.model import CompositeModel
    >>> import numpy as np
    >>>
    >>> # Create a composite model
    >>> gauss1 = GaussianModel(prefix='g1_')
    >>> gauss2 = GaussianModel(prefix='g2_')
    >>> composite = CompositeModel(gauss1, gauss2, op=np.add)
    >>>
    >>> # Generate synthetic data
    >>> x = np.linspace(0, 10, 100)
    >>> y = gauss1.eval(params=gauss1.make_params(amplitude=1, center=5, sigma=1), x=x) + \
    >>>     gauss2.eval(params=gauss2.make_params(amplitude=0.5, center=7, sigma=0.5), x=x)
    >>>
    >>> # Reduce the composite model
    >>> reduced_model = reduce_composite(composite, y, x, r2_threshold=0.98)
    """
    # Ensure the input is a composite model, as only composites can be reduced
    if not isinstance(model, CompositeModel):
        return model, None

    # Make a deep copy of the input model to avoid modifying the original
    model = deepcopy(model)

    # Perform a full fit with the composite model to evaluate its R²
    full_fit = model.fit(y, x=x, params=model.make_params())
    if r2_threshold > full_fit.rsquared:
        # If the full model doesn't meet the R² threshold, return the original model

        return model

    def _reduce_composite(
        _model: CompositeModel, _y: np.ndarray
    ) -> Tuple[CompositeModel, Union[ModelResult, None]]:
        """
        Recursively reduce the components of a composite model by evaluating
        the individual contributions of the left and right sub-models.
        """
        if not isinstance(_model, CompositeModel):
            return _model, None

        # Access the left and right sub-models of the composite
        left = _model.left
        right = _model.right

        # Evaluate the contribution of the left and right models using the full-fit parameters
        left_eval = left.eval(full_fit.params, x=x)
        right_eval = right.eval(full_fit.params, x=x)

        left_fit = left.fit(_y, x=x, params=full_fit.params)
        right_fit = right.fit(_y, x=x, params=full_fit.params)
        left_fit_eval = left_fit.eval(x=x)
        right_fit_eval = right_fit.eval(x=x)
        # Compute the total variance of the current data
        _std = np.sum((_y - np.mean(_y)) ** 2)

        # Compute R² for the left sub-model
        r2_left = 1 - np.sum((_y - left_fit_eval) ** 2) / _std
        if r2_left > r2_threshold and not isinstance(right, CompositeModel):
            # If the left model alone is sufficient, recursively reduce the left
            for param in left_fit.params:
                left.set_param_hint(param, value=left_fit.params[param].value)
                full_fit.params[param].value = left_fit.params[param].value
            return _reduce_composite(left, _y)

        # Compute R² for the right sub-model
        r2_right = 1 - np.sum((_y - right_fit_eval) ** 2) / _std
        if r2_right > r2_threshold and not isinstance(left, CompositeModel):
            # If the right model alone is sufficient, recursively reduce the right
            for param in right_fit.params:
                right.set_param_hint(param, value=right_fit.params[param].value)
                full_fit.params[param].value = right_fit.params[param].value
            return _reduce_composite(right, _y)

        # If neither side is sufficient alone, attempt to reduce each side recursively
        if isinstance(left, CompositeModel):
            left, _ = _reduce_composite(left, _y - right_eval)
        if isinstance(right, CompositeModel):
            right, _ = _reduce_composite(right, _y - left_eval)

        # Reassemble the composite model from the reduced components
        reduced = CompositeModel(left, right, op=_model.op)

        # Refit the reduced composite model to adjust parameters
        reduced_fit = reduced.fit(_y, x=x, params=full_fit.params)
        for param in reduced_fit.params:
            reduced.set_param_hint(param, value=reduced_fit.params[param].value)

        return reduced, reduced_fit

    # Reduce the input model by pruning unnecessary components
    _red = _reduce_composite(
        model,
        y,
    )

    return _red
