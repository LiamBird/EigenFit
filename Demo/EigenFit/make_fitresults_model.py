def make_model(peaks, signs=None, background="linear", center_tolerance=10, peaks_init=0):
    import numpy as np
    from lmfit import Parameters
    from lmfit.models import LinearModel, LorentzianModel, PolynomialModel
    import re
    
    if background=="linear":
        model = LinearModel()
    else:
        model = PolynomialModel(degree=4)
    
    for peak in peaks:
        model += LorentzianModel(prefix="peak_{}_".format(peak))
    params = model.make_params()

    for name in params:
        if "center" in name:
            center_value = int(re.findall("\d+", name)[0])
            params[name].init_value = center_value
            params[name].value = center_value
            if center_tolerance != 0:
                params[name].min = center_value-center_tolerance
                params[name].max = center_value+center_tolerance
            elif center_tolerance == 0:
                params[name].vary = False
            
        if "sigma" in name:
            params[name].min = 1
            params[name].max = 20
            params[name].init_value = 10
            params[name].value = 10

        if "amplitude" in name:
            params[name].init_value = peaks_init
            
            if signs != None:
                if type(signs) == str:
                    if signs == "positive":
                        for peak in peaks:
                            params["peak_{}_amplitude".format(peak)].min = 0
                if type(signs) == list:
                    for nsign, sign in enumerate(signs):
                        if sign == "positive":
                            params["peak_{}_amplitude".format(peaks[nsign])].min = 0
        if "peak" not in name:
            params[name].value = 0
    return model, params