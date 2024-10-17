from typing import Optional, Tuple
import funcnodes as fn
from lmfit.model import Model, Parameters
import numpy as np


def _update_param_entries(src: fn.NodeIO, result: str):
    node = src.node
    if not node:
        return
    model: Model = node.inputs["model"].value
    if not hasattr(model, "param_hints"):
        return

    if result not in model.param_hints:
        return
    v = model.param_hints[result]

    for k in ["value", "min", "max"]:
        if node.inputs[k].is_connected():
            continue
        if k in v:
            node.inputs[k].set_value(v[k], does_trigger=False)
        else:
            node.inputs[k].set_value(fn.NoValue, does_trigger=False)


@fn.NodeDecorator(
    "lmfit.update_model_param",
    description="Update a parameter of a lmfit Model",
    outputs=[{"type": Model, "name": "updated_model"}],
    default_io_options={
        "model": {
            "on": {
                "after_set_value": fn.decorator.update_other_io(
                    "param_name",
                    lambda x: list(x.param_names),
                )
            }
        },
        "param_name": {"on": {"after_set_value": _update_param_entries}},
    },
)
def update_model_param(
    model: Model,
    param_name: str,
    value: Optional[float] = None,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> Model:
    if param_name not in model.param_names:
        raise ValueError(f"Model has no parameter named {param_name}")

    updates = {}
    if value is not None:
        updates["value"] = value
    if min is not None:
        updates["min"] = min
    if max is not None:
        updates["max"] = max

    model.set_param_hint(param_name, **updates)

    return model


@fn.NodeDecorator(
    "lmfit.guess_model",
    description="Guess the parameters of a lmfit Model",
    outputs=[
        {"type": Model, "name": "updated_model"},
        {"type": Parameters, "name": "parameters"},
    ],
)
def guess_model(
    model: Model, y: np.ndarray, x: np.ndarray, update: bool = True
) -> Tuple[Model, Parameters]:
    guess_result = model.guess(y, x=x)

    if update:
        for param_name in guess_result:
            model.set_param_hint(param_name, value=guess_result[param_name].value)

    return model, guess_result


PARAMS_SHELF = fn.Shelf(
    nodes=[update_model_param, guess_model],
    name="Parameters",
    description="Nodes working with lmfit Model parameters",
    subshelves=[],
)
