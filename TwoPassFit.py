import numpy as np
import os

from datetime import datetime
from lmfit.models import LinearModel, LorentzianModel
from tqdm import notebook

from make_lorentzian_params import *

def xy_2PF(x, y, label, SeriesDimensionReduction_obj, SeriesLCF_obj, noise_tolerance=1):
    
    start_time = datetime.now()
    
    spectrum = SeriesDimensionReduction_obj.map_normalised[label][x, y]
    shift = SeriesDimensionReduction_obj.common.shift
    
    neg_idx = np.nonzero(np.sign(spectrum)==-1)
    negative_values = spectrum[neg_idx]

    all_peaks_to_fit = np.unique([item for sublist in [*SeriesLCF_obj.peaks_to_fit[label].values()] for item in sublist])
    xy_model = LinearModel()
    xy_peaks = []

    for peak in all_peaks_to_fit:
        if SeriesLCF_obj.scaled_outputs.intensities[label][peak][x, y] > abs(noise_tolerance*np.nanmedian(negative_values)):
            xy_peaks.append(peak)
            xy_model += LorentzianModel(prefix="peak_{}_".format(peak))

    xy_params = make_lorentzian_params(xy_model)
    xy_fit = xy_model.fit(x=shift, data=spectrum, params=xy_params)
    end_time = datetime.now()
    fitting_time = 1e6*(end_time-start_time).seconds+(end_time-start_time).microseconds
    
    return xy_fit, xy_peaks, fitting_time

class TwoPassFit(object):
    def __init__(self, SeriesDimensionReduction_obj, SeriesLCF_obj, save_path, noise_tolerance=1):
        from datetime import datetime
        from tqdm import notebook
        
        x_extent = SeriesDimensionReduction_obj.common.x_extent
        y_extent = SeriesDimensionReduction_obj.common.y_extent
        shift_extent = SeriesDimensionReduction_obj.common.shift_extent

        self._series_label = SeriesDimensionReduction_obj._series_label
        setattr(self, self._series_label, vars(SeriesDimensionReduction_obj)[self._series_label])
        self.fitted_values = {}
        self.fitted_spectra = {}
        self.fit_times = {}
        
        if save_path != None and os.path.isdir(save_path) == False:
            os.makedirs(save_path)
        
        for label in notebook.tqdm(vars(self)[self._series_label]):
            if save_path != None:
                if str(label) in os.listdir(save_path):
                    self.fitted_values.update([(label, np.load(os.path.join(save_path, str(label), "fitted_lorentzian_params.npy"), allow_pickle=True).item())])
                    self.fitted_spectra.update([(label, np.load(os.path.join(save_path, str(label), "fitted_spectra.npy"), allow_pickle=True))])
                    self.fit_times.update([(label, np.load(os.path.join(save_path, str(label), "time_to_fit.npy"), allow_pickle=True))])
                elif str(label) not in os.listdir(save_path) or all(str(label) in os.listdir(save_path),
                                                  len(os.listdir(os.path.join(save_path), str(label))) != 3):
                    os.makedirs(os.path.join(save_path, str(label)))
                    fitted_values = dict([(peak, dict([(key, 
                                                        np.full((x_extent, y_extent), np.nan))
                                                       for key in ["amplitude", "center", "intensity", "sigma"]]))
                                          for peak in dim_red.selected_peaks[0]._peaks])

                    fitted_spectra = np.full((x_extent, y_extent, shift_extent), np.nan)

                    fitting_times = np.full((x_extent, y_extent), np.nan)

                    for x in range(x_extent):
                        for y in range(y_extent):
                            xy_fit, xy_peaks, xy_time = xy_2PF(x, y, label, 
                                                               SeriesDimensionReduction_obj,
                                                     SeriesLCF_obj, noise_tolerance=noise_tolerance)
                            for peak in xy_peaks:
                                for name in ["amplitude", "center", "sigma"]:
                                    fitted_values[peak][name][x, y] = xy_fit.best_values["peak_{}_{}".format(peak, name)]

                                fitted_values[peak]["intensity"][x, y] = xy_fit.best_values["peak_{}_amplitude".format(peak)]/xy_fit.best_values["peak_{}_sigma".format(peak)]/np.pi

                            fitted_spectra[x, y] = xy_fit.best_fit

                            fitting_times[x, y] = xy_time

                    self.fitted_values.update([(label, fitted_values)])
                    self.fit_times.update([(label, fitting_times)])
                    self.fitted_spectra.update([(label, fitted_spectra)])

                    np.save(os.path.join(save_path, str(label), "fitted_lorentzian_params.npy"),
                            fitted_values, allow_pickle=True)
                    np.save(os.path.join(save_path, str(label), "fitted_spectra.npy"),
                            fitted_spectra, allow_pickle=True)
                    np.save(os.path.join(save_path, str(label), "time_to_fit.npy"),
                            fitting_times, allow_pickle=True)                