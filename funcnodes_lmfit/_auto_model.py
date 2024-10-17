from lmfit.model import CompositeModel, Model
from .utils import autoprefix, update_model_params, model_composit_train_create
import numpy as np
import funcnodes as fn
from typing import Optional
from lmfit.models import (
    ConstantModel,
    ComplexConstantModel,
    LinearModel,
    QuadraticModel,
    GaussianModel,
    Gaussian2dModel,
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


@fn.NodeDecorator(
    id="lmfit.models.ConstantModel",
    description="Create or add a Constant model",
    outputs=[{"type": ConstantModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "c_value": {"hidden": True},
        "c_min": {"hidden": True},
        "c_max": {"hidden": True},
        "c_vary": {"hidden": True},
    },
)
def ConstantModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    c_value: float = 0.0,
    c_min: Optional[float] = None,
    c_max: Optional[float] = None,
    c_vary: bool = True,
) -> Model:
    prefix = autoprefix(ConstantModel, composite)
    model = ConstantModel(prefix=prefix)

    model.set_param_hint(
        "c",
        value=c_value,
        min=c_min if c_min is not None else -np.inf,
        max=c_max if c_max is not None else np.inf,
        vary=c_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.ComplexConstantModel",
    description="Create or add a Complex Constant model",
    outputs=[{"type": ComplexConstantModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "re_value": {"hidden": True},
        "re_min": {"hidden": True},
        "re_max": {"hidden": True},
        "re_vary": {"hidden": True},
        "im_value": {"hidden": True},
        "im_min": {"hidden": True},
        "im_max": {"hidden": True},
        "im_vary": {"hidden": True},
    },
)
def ComplexConstantModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    re_value: float = 0.0,
    re_min: Optional[float] = None,
    re_max: Optional[float] = None,
    re_vary: bool = True,
    im_value: float = 0.0,
    im_min: Optional[float] = None,
    im_max: Optional[float] = None,
    im_vary: bool = True,
) -> Model:
    prefix = autoprefix(ComplexConstantModel, composite)
    model = ComplexConstantModel(prefix=prefix)

    model.set_param_hint(
        "re",
        value=re_value,
        min=re_min if re_min is not None else -np.inf,
        max=re_max if re_max is not None else np.inf,
        vary=re_vary,
    )
    model.set_param_hint(
        "im",
        value=im_value,
        min=im_min if im_min is not None else -np.inf,
        max=im_max if im_max is not None else np.inf,
        vary=im_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.LinearModel",
    description="Create or add a Linear model",
    outputs=[{"type": LinearModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "slope_value": {"hidden": True},
        "slope_min": {"hidden": True},
        "slope_max": {"hidden": True},
        "slope_vary": {"hidden": True},
        "intercept_value": {"hidden": True},
        "intercept_min": {"hidden": True},
        "intercept_max": {"hidden": True},
        "intercept_vary": {"hidden": True},
    },
)
def LinearModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    slope_value: float = 1.0,
    slope_min: Optional[float] = None,
    slope_max: Optional[float] = None,
    slope_vary: bool = True,
    intercept_value: float = 0.0,
    intercept_min: Optional[float] = None,
    intercept_max: Optional[float] = None,
    intercept_vary: bool = True,
) -> Model:
    prefix = autoprefix(LinearModel, composite)
    model = LinearModel(prefix=prefix)

    model.set_param_hint(
        "slope",
        value=slope_value,
        min=slope_min if slope_min is not None else -np.inf,
        max=slope_max if slope_max is not None else np.inf,
        vary=slope_vary,
    )
    model.set_param_hint(
        "intercept",
        value=intercept_value,
        min=intercept_min if intercept_min is not None else -np.inf,
        max=intercept_max if intercept_max is not None else np.inf,
        vary=intercept_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.QuadraticModel",
    description="Create or add a Quadratic model",
    outputs=[{"type": QuadraticModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "a_value": {"hidden": True},
        "a_min": {"hidden": True},
        "a_max": {"hidden": True},
        "a_vary": {"hidden": True},
        "b_value": {"hidden": True},
        "b_min": {"hidden": True},
        "b_max": {"hidden": True},
        "b_vary": {"hidden": True},
        "c_value": {"hidden": True},
        "c_min": {"hidden": True},
        "c_max": {"hidden": True},
        "c_vary": {"hidden": True},
    },
)
def QuadraticModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    a_value: float = 0.0,
    a_min: Optional[float] = None,
    a_max: Optional[float] = None,
    a_vary: bool = True,
    b_value: float = 0.0,
    b_min: Optional[float] = None,
    b_max: Optional[float] = None,
    b_vary: bool = True,
    c_value: float = 0.0,
    c_min: Optional[float] = None,
    c_max: Optional[float] = None,
    c_vary: bool = True,
) -> Model:
    prefix = autoprefix(QuadraticModel, composite)
    model = QuadraticModel(prefix=prefix)

    model.set_param_hint(
        "a",
        value=a_value,
        min=a_min if a_min is not None else -np.inf,
        max=a_max if a_max is not None else np.inf,
        vary=a_vary,
    )
    model.set_param_hint(
        "b",
        value=b_value,
        min=b_min if b_min is not None else -np.inf,
        max=b_max if b_max is not None else np.inf,
        vary=b_vary,
    )
    model.set_param_hint(
        "c",
        value=c_value,
        min=c_min if c_min is not None else -np.inf,
        max=c_max if c_max is not None else np.inf,
        vary=c_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.GaussianModel",
    description="Create or add a Gaussian model",
    outputs=[{"type": GaussianModel, "name": "model"}],
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
def GaussianModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
) -> Model:
    prefix = autoprefix(GaussianModel, composite)
    model = GaussianModel(prefix=prefix)

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
    id="lmfit.models.Gaussian2dModel",
    description="Create or add a Gaussian-2D model",
    outputs=[{"type": Gaussian2dModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "amplitude_value": {"hidden": True},
        "amplitude_min": {"hidden": True},
        "amplitude_max": {"hidden": True},
        "amplitude_vary": {"hidden": True},
        "centerx_value": {"hidden": True},
        "centerx_min": {"hidden": True},
        "centerx_max": {"hidden": True},
        "centerx_vary": {"hidden": True},
        "centery_value": {"hidden": True},
        "centery_min": {"hidden": True},
        "centery_max": {"hidden": True},
        "centery_vary": {"hidden": True},
        "sigmax_value": {"hidden": True},
        "sigmax_min": {"hidden": True},
        "sigmax_max": {"hidden": True},
        "sigmax_vary": {"hidden": True},
        "sigmay_value": {"hidden": True},
        "sigmay_min": {"hidden": True},
        "sigmay_max": {"hidden": True},
        "sigmay_vary": {"hidden": True},
    },
)
def Gaussian2dModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    centerx_value: float = 0.0,
    centerx_min: Optional[float] = None,
    centerx_max: Optional[float] = None,
    centerx_vary: bool = True,
    centery_value: float = 0.0,
    centery_min: Optional[float] = None,
    centery_max: Optional[float] = None,
    centery_vary: bool = True,
    sigmax_value: float = 1.0,
    sigmax_min: Optional[float] = 0,
    sigmax_max: Optional[float] = None,
    sigmax_vary: bool = True,
    sigmay_value: float = 1.0,
    sigmay_min: Optional[float] = 0,
    sigmay_max: Optional[float] = None,
    sigmay_vary: bool = True,
) -> Model:
    prefix = autoprefix(Gaussian2dModel, composite)
    model = Gaussian2dModel(prefix=prefix)

    model.set_param_hint(
        "amplitude",
        value=amplitude_value,
        min=amplitude_min if amplitude_min is not None else -np.inf,
        max=amplitude_max if amplitude_max is not None else np.inf,
        vary=amplitude_vary,
    )
    model.set_param_hint(
        "centerx",
        value=centerx_value,
        min=centerx_min if centerx_min is not None else -np.inf,
        max=centerx_max if centerx_max is not None else np.inf,
        vary=centerx_vary,
    )
    model.set_param_hint(
        "centery",
        value=centery_value,
        min=centery_min if centery_min is not None else -np.inf,
        max=centery_max if centery_max is not None else np.inf,
        vary=centery_vary,
    )
    model.set_param_hint(
        "sigmax",
        value=sigmax_value,
        min=sigmax_min if sigmax_min is not None else -np.inf,
        max=sigmax_max if sigmax_max is not None else np.inf,
        vary=sigmax_vary,
    )
    model.set_param_hint(
        "sigmay",
        value=sigmay_value,
        min=sigmay_min if sigmay_min is not None else -np.inf,
        max=sigmay_max if sigmay_max is not None else np.inf,
        vary=sigmay_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.LorentzianModel",
    description="Create or add a Lorentzian model",
    outputs=[{"type": LorentzianModel, "name": "model"}],
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
def LorentzianModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
) -> Model:
    prefix = autoprefix(LorentzianModel, composite)
    model = LorentzianModel(prefix=prefix)

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
    id="lmfit.models.SplitLorentzianModel",
    description="Create or add a Split-Lorentzian model",
    outputs=[{"type": SplitLorentzianModel, "name": "model"}],
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
        "sigma_r_value": {"hidden": True},
        "sigma_r_min": {"hidden": True},
        "sigma_r_max": {"hidden": True},
        "sigma_r_vary": {"hidden": True},
    },
)
def SplitLorentzianModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    sigma_r_value: float = 1.0,
    sigma_r_min: Optional[float] = 0,
    sigma_r_max: Optional[float] = None,
    sigma_r_vary: bool = True,
) -> Model:
    prefix = autoprefix(SplitLorentzianModel, composite)
    model = SplitLorentzianModel(prefix=prefix)

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
    model.set_param_hint(
        "sigma_r",
        value=sigma_r_value,
        min=sigma_r_min if sigma_r_min is not None else -np.inf,
        max=sigma_r_max if sigma_r_max is not None else np.inf,
        vary=sigma_r_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.VoigtModel",
    description="Create or add a Voigt model",
    outputs=[{"type": VoigtModel, "name": "model"}],
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
def VoigtModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
) -> Model:
    prefix = autoprefix(VoigtModel, composite)
    model = VoigtModel(prefix=prefix)

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
    id="lmfit.models.PseudoVoigtModel",
    description="Create or add a PseudoVoigt model",
    outputs=[{"type": PseudoVoigtModel, "name": "model"}],
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
        "fraction_value": {"hidden": True},
        "fraction_min": {"hidden": True},
        "fraction_max": {"hidden": True},
        "fraction_vary": {"hidden": True},
    },
)
def PseudoVoigtModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    fraction_value: float = 0.5,
    fraction_min: Optional[float] = 0.0,
    fraction_max: Optional[float] = 1.0,
    fraction_vary: bool = True,
) -> Model:
    prefix = autoprefix(PseudoVoigtModel, composite)
    model = PseudoVoigtModel(prefix=prefix)

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
    model.set_param_hint(
        "fraction",
        value=fraction_value,
        min=fraction_min if fraction_min is not None else -np.inf,
        max=fraction_max if fraction_max is not None else np.inf,
        vary=fraction_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.MoffatModel",
    description="Create or add a Moffat model",
    outputs=[{"type": MoffatModel, "name": "model"}],
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
        "beta_value": {"hidden": True},
        "beta_min": {"hidden": True},
        "beta_max": {"hidden": True},
        "beta_vary": {"hidden": True},
    },
)
def MoffatModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    beta_value: float = 1.0,
    beta_min: Optional[float] = None,
    beta_max: Optional[float] = None,
    beta_vary: bool = True,
) -> Model:
    prefix = autoprefix(MoffatModel, composite)
    model = MoffatModel(prefix=prefix)

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
    model.set_param_hint(
        "beta",
        value=beta_value,
        min=beta_min if beta_min is not None else -np.inf,
        max=beta_max if beta_max is not None else np.inf,
        vary=beta_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.Pearson4Model",
    description="Create or add a Pearson4 model",
    outputs=[{"type": Pearson4Model, "name": "model"}],
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
        "expon_value": {"hidden": True},
        "expon_min": {"hidden": True},
        "expon_max": {"hidden": True},
        "expon_vary": {"hidden": True},
        "skew_value": {"hidden": True},
        "skew_min": {"hidden": True},
        "skew_max": {"hidden": True},
        "skew_vary": {"hidden": True},
    },
)
def Pearson4Model_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    expon_value: float = 1.5,
    expon_min: Optional[float] = 0.500000000000001,
    expon_max: Optional[float] = 1000,
    expon_vary: bool = True,
    skew_value: float = 0.0,
    skew_min: Optional[float] = -1000,
    skew_max: Optional[float] = 1000,
    skew_vary: bool = True,
) -> Model:
    prefix = autoprefix(Pearson4Model, composite)
    model = Pearson4Model(prefix=prefix)

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
    model.set_param_hint(
        "expon",
        value=expon_value,
        min=expon_min if expon_min is not None else -np.inf,
        max=expon_max if expon_max is not None else np.inf,
        vary=expon_vary,
    )
    model.set_param_hint(
        "skew",
        value=skew_value,
        min=skew_min if skew_min is not None else -np.inf,
        max=skew_max if skew_max is not None else np.inf,
        vary=skew_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.Pearson7Model",
    description="Create or add a Pearson7 model",
    outputs=[{"type": Pearson7Model, "name": "model"}],
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
        "expon_value": {"hidden": True},
        "expon_min": {"hidden": True},
        "expon_max": {"hidden": True},
        "expon_vary": {"hidden": True},
    },
)
def Pearson7Model_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    expon_value: float = 1.5,
    expon_min: Optional[float] = None,
    expon_max: Optional[float] = 100,
    expon_vary: bool = True,
) -> Model:
    prefix = autoprefix(Pearson7Model, composite)
    model = Pearson7Model(prefix=prefix)

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
    model.set_param_hint(
        "expon",
        value=expon_value,
        min=expon_min if expon_min is not None else -np.inf,
        max=expon_max if expon_max is not None else np.inf,
        vary=expon_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.StudentsTModel",
    description="Create or add a StudentsT model",
    outputs=[{"type": StudentsTModel, "name": "model"}],
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
def StudentsTModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0.0,
    sigma_max: Optional[float] = 100,
    sigma_vary: bool = True,
) -> Model:
    prefix = autoprefix(StudentsTModel, composite)
    model = StudentsTModel(prefix=prefix)

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
    id="lmfit.models.BreitWignerModel",
    description="Create or add a Breit-Wigner model",
    outputs=[{"type": BreitWignerModel, "name": "model"}],
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
        "q_value": {"hidden": True},
        "q_min": {"hidden": True},
        "q_max": {"hidden": True},
        "q_vary": {"hidden": True},
    },
)
def BreitWignerModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0.0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    q_value: float = 1.0,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
    q_vary: bool = True,
) -> Model:
    prefix = autoprefix(BreitWignerModel, composite)
    model = BreitWignerModel(prefix=prefix)

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
    model.set_param_hint(
        "q",
        value=q_value,
        min=q_min if q_min is not None else -np.inf,
        max=q_max if q_max is not None else np.inf,
        vary=q_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.LognormalModel",
    description="Create or add a Log-Normal model",
    outputs=[{"type": LognormalModel, "name": "model"}],
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
def LognormalModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
) -> Model:
    prefix = autoprefix(LognormalModel, composite)
    model = LognormalModel(prefix=prefix)

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
    id="lmfit.models.DampedOscillatorModel",
    description="Create or add a Damped Oscillator model",
    outputs=[{"type": DampedOscillatorModel, "name": "model"}],
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
def DampedOscillatorModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 1.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 0.1,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
) -> Model:
    prefix = autoprefix(DampedOscillatorModel, composite)
    model = DampedOscillatorModel(prefix=prefix)

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
    id="lmfit.models.DampedHarmonicOscillatorModel",
    description="Create or add a Damped Harmonic Oscillator model",
    outputs=[{"type": DampedHarmonicOscillatorModel, "name": "model"}],
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
        "gamma_value": {"hidden": True},
        "gamma_min": {"hidden": True},
        "gamma_max": {"hidden": True},
        "gamma_vary": {"hidden": True},
    },
)
def DampedHarmonicOscillatorModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 1.0,
    center_min: Optional[float] = 0,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    gamma_value: float = 1.0,
    gamma_min: Optional[float] = 1e-19,
    gamma_max: Optional[float] = None,
    gamma_vary: bool = True,
) -> Model:
    prefix = autoprefix(DampedHarmonicOscillatorModel, composite)
    model = DampedHarmonicOscillatorModel(prefix=prefix)

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
    model.set_param_hint(
        "gamma",
        value=gamma_value,
        min=gamma_min if gamma_min is not None else -np.inf,
        max=gamma_max if gamma_max is not None else np.inf,
        vary=gamma_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.ExponentialGaussianModel",
    description="Create or add a Exponential Gaussian model",
    outputs=[{"type": ExponentialGaussianModel, "name": "model"}],
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
        "gamma_value": {"hidden": True},
        "gamma_min": {"hidden": True},
        "gamma_max": {"hidden": True},
        "gamma_vary": {"hidden": True},
    },
)
def ExponentialGaussianModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    gamma_value: float = 1.0,
    gamma_min: Optional[float] = 0,
    gamma_max: Optional[float] = 20,
    gamma_vary: bool = True,
) -> Model:
    prefix = autoprefix(ExponentialGaussianModel, composite)
    model = ExponentialGaussianModel(prefix=prefix)

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
    model.set_param_hint(
        "gamma",
        value=gamma_value,
        min=gamma_min if gamma_min is not None else -np.inf,
        max=gamma_max if gamma_max is not None else np.inf,
        vary=gamma_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.SkewedGaussianModel",
    description="Create or add a Skewed Gaussian model",
    outputs=[{"type": SkewedGaussianModel, "name": "model"}],
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
        "gamma_value": {"hidden": True},
        "gamma_min": {"hidden": True},
        "gamma_max": {"hidden": True},
        "gamma_vary": {"hidden": True},
    },
)
def SkewedGaussianModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    gamma_value: float = 0.0,
    gamma_min: Optional[float] = None,
    gamma_max: Optional[float] = None,
    gamma_vary: bool = True,
) -> Model:
    prefix = autoprefix(SkewedGaussianModel, composite)
    model = SkewedGaussianModel(prefix=prefix)

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
    model.set_param_hint(
        "gamma",
        value=gamma_value,
        min=gamma_min if gamma_min is not None else -np.inf,
        max=gamma_max if gamma_max is not None else np.inf,
        vary=gamma_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.SkewedVoigtModel",
    description="Create or add a Skewed Voigt model",
    outputs=[{"type": SkewedVoigtModel, "name": "model"}],
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
        "skew_value": {"hidden": True},
        "skew_min": {"hidden": True},
        "skew_max": {"hidden": True},
        "skew_vary": {"hidden": True},
    },
)
def SkewedVoigtModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0.0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = 0,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    skew_value: float = 0.0,
    skew_min: Optional[float] = None,
    skew_max: Optional[float] = None,
    skew_vary: bool = True,
) -> Model:
    prefix = autoprefix(SkewedVoigtModel, composite)
    model = SkewedVoigtModel(prefix=prefix)

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
    model.set_param_hint(
        "skew",
        value=skew_value,
        min=skew_min if skew_min is not None else -np.inf,
        max=skew_max if skew_max is not None else np.inf,
        vary=skew_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.DoniachModel",
    description="Create or add a Doniach model",
    outputs=[{"type": DoniachModel, "name": "model"}],
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
        "gamma_value": {"hidden": True},
        "gamma_min": {"hidden": True},
        "gamma_max": {"hidden": True},
        "gamma_vary": {"hidden": True},
    },
)
def DoniachModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1.0,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    center_value: float = 0,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    center_vary: bool = True,
    sigma_value: float = 1.0,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    sigma_vary: bool = True,
    gamma_value: float = 0.0,
    gamma_min: Optional[float] = None,
    gamma_max: Optional[float] = None,
    gamma_vary: bool = True,
) -> Model:
    prefix = autoprefix(DoniachModel, composite)
    model = DoniachModel(prefix=prefix)

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
    model.set_param_hint(
        "gamma",
        value=gamma_value,
        min=gamma_min if gamma_min is not None else -np.inf,
        max=gamma_max if gamma_max is not None else np.inf,
        vary=gamma_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.PowerLawModel",
    description="Create or add a Power Law model",
    outputs=[{"type": PowerLawModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "amplitude_value": {"hidden": True},
        "amplitude_min": {"hidden": True},
        "amplitude_max": {"hidden": True},
        "amplitude_vary": {"hidden": True},
        "exponent_value": {"hidden": True},
        "exponent_min": {"hidden": True},
        "exponent_max": {"hidden": True},
        "exponent_vary": {"hidden": True},
    },
)
def PowerLawModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    exponent_value: float = 1.0,
    exponent_min: Optional[float] = None,
    exponent_max: Optional[float] = None,
    exponent_vary: bool = True,
) -> Model:
    prefix = autoprefix(PowerLawModel, composite)
    model = PowerLawModel(prefix=prefix)

    model.set_param_hint(
        "amplitude",
        value=amplitude_value,
        min=amplitude_min if amplitude_min is not None else -np.inf,
        max=amplitude_max if amplitude_max is not None else np.inf,
        vary=amplitude_vary,
    )
    model.set_param_hint(
        "exponent",
        value=exponent_value,
        min=exponent_min if exponent_min is not None else -np.inf,
        max=exponent_max if exponent_max is not None else np.inf,
        vary=exponent_vary,
    )

    return model_composit_train_create(model, composite, x, y)


@fn.NodeDecorator(
    id="lmfit.models.ExponentialModel",
    description="Create or add a Exponential model",
    outputs=[{"type": ExponentialModel, "name": "model"}],
    default_io_options={
        "model": {"on": {"after_set_value": update_model_params()}},
        "amplitude_value": {"hidden": True},
        "amplitude_min": {"hidden": True},
        "amplitude_max": {"hidden": True},
        "amplitude_vary": {"hidden": True},
        "decay_value": {"hidden": True},
        "decay_min": {"hidden": True},
        "decay_max": {"hidden": True},
        "decay_vary": {"hidden": True},
    },
)
def ExponentialModel_node(
    composite: Optional[CompositeModel] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    amplitude_value: float = 1,
    amplitude_min: Optional[float] = None,
    amplitude_max: Optional[float] = None,
    amplitude_vary: bool = True,
    decay_value: float = 1,
    decay_min: Optional[float] = None,
    decay_max: Optional[float] = None,
    decay_vary: bool = True,
) -> Model:
    prefix = autoprefix(ExponentialModel, composite)
    model = ExponentialModel(prefix=prefix)

    model.set_param_hint(
        "amplitude",
        value=amplitude_value,
        min=amplitude_min if amplitude_min is not None else -np.inf,
        max=amplitude_max if amplitude_max is not None else np.inf,
        vary=amplitude_vary,
    )
    model.set_param_hint(
        "decay",
        value=decay_value,
        min=decay_min if decay_min is not None else -np.inf,
        max=decay_max if decay_max is not None else np.inf,
        vary=decay_vary,
    )

    return model_composit_train_create(model, composite, x, y)


_AUTOMODELS = [
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
]
