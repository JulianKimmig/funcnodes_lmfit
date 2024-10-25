import funcnodes as fn
from .model import MODEL_SHELF
from .params import PARAMS_SHELF
from .plotting import PLOT_SHELF
from .fitting import FIT_SHELF
from .use_model import USEMODEL_SHELF

__version__ = "0.1.2"

NODE_SHELF = fn.Shelf(
    nodes=[],
    name="Funcnodes Lmfit",
    description="The nodes for the lmfit package",
    subshelves=[USEMODEL_SHELF, MODEL_SHELF, PARAMS_SHELF, PLOT_SHELF, FIT_SHELF],
)
