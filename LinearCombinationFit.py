import numpy as np
import os

from lmfit import Parameters
from lmfit.models import LinearModel, LorentzianModel

from make_lorentzian_params import *

def linear_combination_fit(shift, components, selected_peaks, center_tolerance=10):
    from lmfit import Parameters
    from lmfit.models import LinearModel, LorentzianModel
    
    peaks_to_fit = {}
    models_to_fit = {}
    fitted_models = {}
    
    n_components = selected_peaks.shape[0]
    for n in range(n_components):
        model = LinearModel()
        peaks = [keys for keys, values in selected_peaks[n][0].items() if values==True]
        
        for peak in peaks:
            model += LorentzianModel(prefix="peak_{}_".format(peak))
        
        params = make_lorentzian_params(model)
        
        fit_out = model.fit(x=shift, data=components[n], params=params)
            
        peaks_to_fit.update([(n, peaks)])
        models_to_fit.update([(n, model)])
        fitted_models.update([(n, fit_out)])
        
    return peaks_to_fit, models_to_fit, fitted_models

class SeriesLCF(object):
    def __init__(self, SeriesDimensionReduction_obj):
        
        from tqdm import notebook
        
        self._series_label = SeriesDimensionReduction_obj._series_label
        setattr(self, self._series_label, vars(SeriesDimensionReduction_obj)[SeriesDimensionReduction_obj._series_label])
        self.common = SeriesDimensionReduction_obj.common
        
        self._all_peaks = SeriesDimensionReduction_obj.selected_peaks[0]._peaks
        
        n_components = len(SeriesDimensionReduction_obj.selected_peaks)
        self.fitted_models = {}
        self.peaks_to_fit = {}
        self.models_to_fit = {}
        
        for label in notebook.tqdm(vars(self)[self._series_label]):
            components = SeriesDimensionReduction_obj.pca_outputs.components[label][:n_components, :]
            selected_peaks =np.vstack((values.saved_values[label] for values in SeriesDimensionReduction_obj.selected_peaks))
            peaks_to_fit, models_to_fit, fitted_models = linear_combination_fit(shift=self.common.shift,
                                                                                components=components,
                                                                                selected_peaks=selected_peaks)
            
        
            self.fitted_models.update([(label, fitted_models)])
            self.peaks_to_fit.update([(label, peaks_to_fit)])
            self.models_to_fit.update([(label, models_to_fit)])
            
        class _LCFOutputs(object):
            def __init__(lcf_self, labels, all_peaks, n_components):
                keys = ["amplitudes", "sigmas", "centers", "intensities"]
                lcf_self.__dict__.update([(key, 
                                       dict([(label, 
                                              dict([(peak, np.zeros((n_components), dtype=float)) for peak in all_peaks]))
                                             for label in labels]))
                                      for key in keys])
            
        setattr(self, "LCF_outputs", _LCFOutputs(vars(self)[self._series_label], self._all_peaks, n_components))
        
        for label in vars(self)[self._series_label]:
            for n in range(n_components):
                for peak in self.peaks_to_fit[label][n]:
                    self.LCF_outputs.intensities[label][peak][n] = self.fitted_models[label][n].best_values["peak_{}_amplitude".format(peak)]/self.fitted_models[label][n].best_values["peak_{}_sigma".format(peak)]/np.pi
                    self.LCF_outputs.amplitudes[label][peak][n] = self.fitted_models[label][n].best_values["peak_{}_amplitude".format(peak)]
                    self.LCF_outputs.sigmas[label][peak][n] = self.fitted_models[label][n].best_values["peak_{}_sigma".format(peak)]
                    self.LCF_outputs.centers[label][peak][n] = self.fitted_models[label][n].best_values["peak_{}_center".format(peak)]
       
        class _ScaledOutputs(object):
            def __init__(scaled_self, labels, all_peaks):
                keys = ["amplitudes", "_sigmas", "intensities"]
                scaled_self.__dict__.update([(key, 
                                       dict([(label,
                                               dict([(peak, None)
                                                     for peak in all_peaks]))
                                            for label in labels]))
                                            for key in keys])
                
        setattr(self, "scaled_outputs", _ScaledOutputs(vars(self)[self._series_label], self._all_peaks))
        
        for label in vars(self)[self._series_label]:
            for peak in self._all_peaks:
                self.scaled_outputs.intensities[label][peak] = np.dot(SeriesDimensionReduction_obj.pca_outputs.scores[label][:, :n_components],
                                                                self.LCF_outputs.intensities[label][peak]).reshape((self.common.x_extent, self.common.y_extent))
            
                self.scaled_outputs.amplitudes[label][peak] = np.dot(SeriesDimensionReduction_obj.pca_outputs.scores[label][:, :n_components],
                                                                self.LCF_outputs.amplitudes[label][peak]).reshape((self.common.x_extent, self.common.y_extent))
                
                self.scaled_outputs._sigmas[label][peak] = np.dot(SeriesDimensionReduction_obj.pca_outputs.scores[label][:, :n_components],
                                                                self.LCF_outputs.sigmas[label][peak]).reshape((self.common.x_extent, self.common.y_extent))