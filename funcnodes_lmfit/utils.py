from typing import Type, Union, Optional
from lmfit.model import Model, CompositeModel
import funcnodes as fn
import numpy as np


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
                composite_model_fit = composite.fit(data=y, x=x)
                y_res = y - composite_model_fit.eval(x=x)
            else:
                y_res = y
            try:
                params = model.guess(y_res, x=x)
            except Exception:
                params = model.make_params()

            try:
                fit_res = model.fit(y_res, params=params, x=x)
                for param in fit_res.params:
                    model.set_param_hint(param, value=fit_res.params[param].value)
            except Exception:
                pass

    if composite is not None:
        model = composite + model

    model._current_x = x
    model._current_y = y
    return model
