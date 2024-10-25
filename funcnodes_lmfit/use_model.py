import funcnodes as fn
from typing import Tuple
from lmfit.model import Model

import numpy as np

try:
    import pandas as pd

    PANDAS_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    PANDAS_INSTALLED = False


@fn.NodeDecorator(
    "lmfit.predict",
    description="Use a lmfit Model to fit data",
    outputs=[
        {"name": "prediction"},
        {"name": "components"},
    ],
)
def predict(
    x: np.ndarray,
    model: Model,
) -> Tuple[np.ndarray, np.ndarray]:
    results = [
        v for k, v in model.eval_components(x=x, params=model.make_params()).items()
    ]

    # merge results to an NxC array where N is the number of data points and C is the number of components#
    merged_results = np.vstack(results).T

    return merged_results.sum(axis=1), merged_results


if PANDAS_INSTALLED:

    @fn.NodeDecorator(
        "lmfit.predict_df",
        description="Use a lmfit Model to fit data and return a DataFrame",
        outputs=[
            {"name": "predictions"},
        ],
    )
    def predict_df(
        x: np.ndarray,
        model: Model,
    ) -> pd.DataFrame:
        data = {
            k: v
            for k, v in model.eval_components(x=x, params=model.make_params()).items()
        }
        if len(data) > 1:
            total = np.vstack([v for k, v in data.items()]).T.sum(axis=1)

            data = {
                model.name: total,
                **data,
            }

        return pd.DataFrame(data)


USEMODEL_SHELF = fn.Shelf(
    nodes=[
        predict,
    ]
    + ([predict_df] if PANDAS_INSTALLED else []),
    name="Use Model",
    description="Nodes for using lmfit models",
    subshelves=[],
)
