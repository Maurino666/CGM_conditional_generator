from .metric_utils import *

from .mage import compute_mage
from .agp import compute_agp, plot_agp

from .arx_delta_r2 import (
    compute_arx_delta_r2,
    arx_ab_test_same_sample,
    compute_arx_delta_r2_over_horizons,
    plot_AB_arx_delta_r2
)

from .granger_block_f_test import compute_granger_block_over_horizons
from .granger_decomposition import (
    compute_granger_decomposition_over_horizons,
    plot_granger_decomposition,
    plot_AB_granger_decomposition
)
from .delta_r2_nonlinear import (
    compute_delta_r2_cv_nonlinear,
    compute_delta_r2_cv_nonlinear_ab,
    plot_delta_r2_cv,
    TemporalCVSpec
)
from .cmi import (
    cmi_decomposition_over_horizons,
    plot_cmi_decomposition
)

from .partial_dcor import compute_partial_dcor


__all__ = [
    # metric_utils
    "compute_lags",
    "get_valid_segments",
    "standardize_series",
    "align_data",

    "build_lagged_view",
    "build_future_targets",

    # metrics
    "compute_mage",

    "compute_agp",
    "plot_agp",

    "compute_arx_delta_r2",
    "compute_arx_delta_r2_over_horizons",
    "arx_ab_test_same_sample",
    "plot_AB_arx_delta_r2",

    "compute_granger_block_over_horizons",
    "compute_granger_decomposition_over_horizons",
    "plot_granger_decomposition",
    "plot_AB_granger_decomposition",

    "compute_delta_r2_cv_nonlinear",
    "compute_delta_r2_cv_nonlinear_ab",

    "plot_delta_r2_cv",
    "TemporalCVSpec",

    "cmi_decomposition_over_horizons",
    "plot_cmi_decomposition",

    "compute_partial_dcor",
]