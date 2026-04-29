from lmfit.models import LinearModel, LorentzianModel
from lmfit import Parameters

def make_lorentzian_params(model, center_tolerance=10, sigma_value=2, sigma_max=20):
    params = model.make_params()
    params["slope"].value = 0
    params["intercept"].value = 0
    for name in params:
        if "peak" in name:
            center = int(name.split("_")[1])
        if "center" in name:
            params[name].value = center
            params[name].min = center-center_tolerance
            params[name].max = center+center_tolerance
        if "sigma" in name:
            params[name].value = sigma_value
            params[name].max = sigma_max
            params[name].min = 1
        if "amplitude" in name:
            params[name].min = 0
    return params         
