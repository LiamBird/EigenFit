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
  
  
class FitResults(object):
    def __init__(self, DimRed_object, SelectPeaks_objects, center_tolerance=10, peaks_init=[0, 0, 0]):
        import numpy as np
        from datetime import datetime
        import re
        
        self._DimRed_object = DimRed_object
        
        try:
            from tqdm import notebook
            from notebook import tqdm as notebook
        except:
            from tqdm import tqdm_notebook as notebook
        
        if type(SelectPeaks_objects) == list:
            voltages = SelectPeaks_objects[0]._voltages
            peaks = SelectPeaks_objects[0]._peaks
            [selection._get_positions() for selection in SelectPeaks_objects];
        else:
            voltages = SelectPeaks_objects._voltages
            peaks = SelectPeaks_objects._peaks
            SelectPeaks_objects._get_positions() 
        self.peaks_to_fit = {}

        self.common = DimRed_object.common
        self.n_components = len(SelectPeaks_objects)
        
## Getting the lists of peaks for fit for each voltage and each component
        for nvolt, volt in enumerate(voltages):
            self.peaks_to_fit.update([(volt, {})])
            for ncomp, comp in enumerate(list(SelectPeaks_objects)):
                self.peaks_to_fit[volt].update([("component{}".format(ncomp+1),
                                                 [peak for npeak, peak in enumerate(peaks) if comp._values_grid[nvolt, npeak]==True])])
## Fitting the principal components                
        self.fit_results = dict([(voltage,
                                  dict([(keys, None) for keys in self.peaks_to_fit[voltage]]))
                                 for voltage in voltages])
        self.fit_times = dict([(voltage,
                                  dict([(keys, None) for keys in self.peaks_to_fit[voltage]]))
                                 for voltage in voltages])       
        for voltage in notebook(voltages):
            for ncomponent, component in enumerate(self.peaks_to_fit[voltage].keys()):
                component_model, component_params = make_model(peaks=self.peaks_to_fit[voltage][component],
                                                               center_tolerance=center_tolerance, peaks_init=peaks_init[ncomponent])
                fit_start = datetime.now()
                self.fit_results[voltage][component] = component_model.fit(x=DimRed_object.common.shift,
                                                                           data=DimRed_object.components[voltage][ncomponent],
                                                                          params=component_params)
                fit_end = datetime.now()
                self.fit_times[voltage][component] = fit_end-fit_start
                
## Scaling the fitted component results to the coefficient at each x, y position
        all_peaks = np.unique([item for sublist in [item for sublist in [[*components.values()] for components in self.peaks_to_fit.values()] for item in sublist] for item in sublist])    
        self.component_intensity = dict([(voltage, {}) for voltage in self.peaks_to_fit.keys()])
        for voltage in self.peaks_to_fit.keys():
            for peak in all_peaks:
                self.component_intensity[voltage].update([(peak,
                                                           [0]*len(self.peaks_to_fit[voltage]))])
                for ncomponent, component in enumerate(self.peaks_to_fit[voltage].keys()):
                    if any([str(peak) in key for key in [*self.fit_results[voltage][component].best_values.keys()]]):
                        amplitude = self.fit_results[voltage][component].best_values["peak_{}_amplitude".format(peak)]
                        sigma = self.fit_results[voltage][component].best_values["peak_{}_sigma".format(peak)]
                        self.component_intensity[voltage][peak][ncomponent] = amplitude/np.pi/sigma
        
        self.scaled_results = dict([(peaks, {}) for peaks in all_peaks])
        
        for peak in all_peaks:
            for voltage in self.peaks_to_fit.keys():
                self.scaled_results[peak].update([(voltage,
                                                   np.sum(\
                                                          np.array(\
                                                                   [DimRed_object.fit_transformed[voltage][:, n]*self.component_intensity[voltage][peak][n] \
                                                                   for n in range(len(self.peaks_to_fit[voltage]))]),
                                                         axis=0).reshape(DimRed_object.common.x_extent, DimRed_object.common.y_extent))])
        self.peak_labels = all_peaks               
        self.voltage_labels = [*self.peaks_to_fit.keys()]
        
        self.scaled_fits = {}

        for voltage in self.voltage_labels:
            best_fits = np.array([results.best_fit for results in self.fit_results[voltage].values()]).T
            scalers = DimRed_object.data_reduced[voltage][:, :self.n_components].T
            self.scaled_fits.update([(voltage, np.dot(best_fits, scalers).T)])
            
    def plot_fit_results(self):
        import matplotlib.pyplot as plt
        from ipywidgets import interact, IntSlider
        import numpy as np
        
        f, ax = plt.subplots()
        def update(x, y, v_idx):
            grid_ref = np.arange(self.common.x_extent*self.common.y_extent).reshape(self.common.x_extent, self.common.y_extent)
            p_idx = grid_ref[x, y]
            
            DimRed_object = self._DimRed_object
            
            voltage = self.voltage_labels[v_idx]
            ax.cla()
            ax.set_title("{} V".format(voltage))
            ax.plot(self.common.shift, DimRed_object.normalised[voltage][p_idx])
            ax.plot(self.common.shift, DimRed_object.data_reconstructed[voltage][p_idx])
            ax.plot(self.common.shift, self.scaled_fits[voltage][p_idx])
        interact(update, x=IntSlider(min=0, max=self.common.x_extent-1, step=1),
                         y=IntSlider(min=0, max=self.common.y_extent-1, step=1),
                         v_idx=IntSlider(min=0, max=len(self.voltage_labels)-1))
                 
        return f, ax
        

        
