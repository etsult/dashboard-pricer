from .svi import fit_svi_surface
from .ssvi import fit_ssvi_surface
from .heston import fit_heston_surface

MODEL_REGISTRY = {
    "SVI": fit_svi_surface,
    "SSVI": fit_ssvi_surface,
    "Heston": fit_heston_surface,
}
