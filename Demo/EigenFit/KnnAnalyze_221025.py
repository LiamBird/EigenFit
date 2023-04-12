import matplotlib.pyplot as plt
import numpy as np
import os
import EigenFit as eig
from CycVolt import CycVolt
import re

from lmfit.models import LinearModel, LorentzianModel
from lmfit import Parameters
from datetime import datetime

from ipywidgets import interact, IntSlider
from sklearn.decomposition import PCA 

try:
    from tqdm import notebook
    from notebook import tqdm as notebook
except:
    from tqdm import tqdm_notebook as notebook

class _TwoPass(object):
    def __init__(two_pass_self, self, cycle):
        load_path = os.path.join(self.sample_path, "Pre-initialised_fit", cycle)
        two_pass_self.values = dict([(float(re.findall("\d+.\d+", voltage)[0]),
                                             np.load(os.path.join(load_path, voltage, "fit_results_values.npy"),
                                                     allow_pickle=True).item())
                                           for voltage in os.listdir(os.path.join(load_path))])
        two_pass_self.spectra = dict([(float(re.findall("\d+.\d+", voltage)[0]),
                                             np.load(os.path.join(load_path, voltage, "fit_results_spectra.npy"),
                                                     allow_pickle=True))
                                           for voltage in os.listdir(os.path.join(load_path))])
        
        
class _KNNpeaks(object):
    def __init__(knn_self, self, cycle, x_component=0, y_component=1, n_neighbours=3, random_seed=0, n_training_spectra=100):
        import random
        from sklearn.decomposition import PCA
        
        knn_self._cycle = cycle
        knn_self.x_component = x_component
        knn_self.y_component = y_component
        knn_self.x_extent = self.dim_red[cycle].common.x_extent
        knn_self.y_extent = self.dim_red[cycle].common.y_extent
        knn_self.data_extent = self.dim_red[cycle].common.x_extent*self.dim_red[cycle].common.y_extent
        
        knn_self.n_training_spectra = n_training_spectra
        knn_self.n_neighbours = n_neighbours
        knn_self.merged = np.vstack((self.dim_red[cycle].normalised.values()))
        
        knn_self.pca = PCA()
        knn_self.fit_transformed = knn_self.pca.fit_transform(knn_self.merged)
        knn_self.components = knn_self.pca.components_
        
        random.seed(random_seed)
        knn_self.spectra_list = []
        while len(knn_self.spectra_list) < n_training_spectra:
            new_number = int(knn_self.merged.shape[0]*random.random())
            if new_number not in knn_self.spectra_list:
                knn_self.spectra_list.append(new_number)
            else:
                continue
                
        knn_self._ft = knn_self.fit_transformed
        
        knn_self.features = np.array([[knn_self._ft[idx, knn_self.x_component],
                                       knn_self._ft[idx, knn_self.y_component]] for idx in knn_self.spectra_list])
        
        knn_self.label_lists = {}
        
        knn_self._outer_self = self
        
        
    def peak_select_viewer(knn_self):
        self = knn_self._outer_self
        cycle = knn_self._cycle
        from ipywidgets import interact, IntSlider
        f, (axes) = plt.subplots(1, 2, figsize=(10, 4))
        axes[1].plot(knn_self.fit_transformed[:, knn_self.x_component],
                    knn_self.fit_transformed[:, knn_self.y_component], "o")
        marker, = axes[1].plot([], [], "o")

        def update(idx):
            i = knn_self.spectra_list[idx]
            marker.set_xdata(knn_self.fit_transformed[i, knn_self.x_component])
            marker.set_ydata(knn_self.fit_transformed[i, knn_self.y_component])

            axes[0].cla()
            axes[0].plot(self.dim_red[cycle].common.shift, 
                         knn_self.merged[i])

        interact(update, idx=IntSlider(min=0, max=knn_self.n_training_spectra-1, step=1))
        
    def fit_model(knn_self, label_key):
        from sklearn.neighbors import KNeighborsClassifier
        self = knn_self._outer_self
        
        labels = np.zeros((knn_self.n_training_spectra))
        for idx in knn_self.label_lists[label_key]:
            labels[idx] = 1
        model = KNeighborsClassifier(n_neighbors=knn_self.n_neighbours)
        
        model.fit(knn_self.features, labels)
        
        test_data = [knn_self._ft[idx, [knn_self.x_component, knn_self.y_component]] for idx in range(knn_self._ft.shape[0]) if idx not in knn_self.spectra_list]
        
        test_predict = model.predict(test_data)
        
        ### Why is this necessary?! - need to check why not just
        ## [n for n in np.arange(knn_self.merged.shape[0]) if n not in knn_self.spectra_list]
        all_idx = np.arange(knn_self.merged.shape[0], dtype=float)
        for n in range(all_idx.shape[0]):
            if all_idx[n] in knn_self.spectra_list:
                all_idx[n] = np.nan
        test_idx = [int(n) for n in all_idx if np.isfinite(n)]
        
        merged_predict = np.full((knn_self.merged.shape[0]), np.nan)
        for n, idx in enumerate(knn_self.spectra_list):
            merged_predict[idx] = labels[n]
        for n, idx in enumerate(test_idx):
            merged_predict[idx] = test_predict[n]
        knn_self.predict_dict = dict([(voltage,
                                   merged_predict[n*knn_self.data_extent:(n+1)*knn_self.data_extent].reshape(knn_self.x_extent, knn_self.y_extent))
                                 for n, voltage in enumerate(self.dim_red[knn_self._cycle].voltage)])
        
    def check_fit(knn_self):
        present = np.vstack(([values.flatten() for values in knn_self.predict_dict.values()])).flatten()
        f, ax = plt.subplots()
        ax.plot(knn_self.fit_transformed[:, knn_self.x_component],
                    knn_self.fit_transformed[:, knn_self.y_component], "o")
        ax.plot(knn_self.fit_transformed[np.argwhere(present==1), knn_self.x_component],
                    knn_self.fit_transformed[np.argwhere(present==1), knn_self.y_component], "o")
        return f, ax
    
    
class CellData(object):
    def __init__(self, sample_path, defaults=True, fit_results=True):
        self.sample_path = sample_path
        self.raw_data = dict([(cycle_label, eig.MapSeries(os.path.join(sample_path, "RawData", cycle_label))) 
                                for cycle_label in os.listdir(os.path.join(sample_path, "RawData")) if "charge" in cycle_label])
        
        if defaults == True:
            try:
                self.CRE = dict([(cycle_label.strip(".npy"), 
                     np.load(os.path.join(sample_path, "CRE_manual_selection", cycle_label), allow_pickle=True).item())
                    for cycle_label in os.listdir(os.path.join(sample_path, "CRE_manual_selection")) if "charge" in cycle_label])

                for cycle_label, CRE_values in self.CRE.items():
                    for n in range(len(CRE_values["voltage"])):
                        self.raw_data[cycle_label].data[CRE_values["voltage"][n]].fix_cosmic_ray(x_pos=CRE_values["x_pos"][n],
                                                                                            y_pos=CRE_values["y_pos"][n],
                                                                                            spike_pos=CRE_values["spike_pos"][n],
                                                                                            smooth_width=CRE_values["smooth_width"][n])
            except:
                pass
            [map_series.set_clip_range(100, 1000) for map_series in self.raw_data.values()];
            
        
            for cycle_label in self.raw_data.keys():
                if os.path.isdir(os.path.join(sample_path, "processed_data", cycle_label)) == False:
                    os.makedirs(os.path.join(sample_path, "processed_data", cycle_label))
                    
            self.dim_red = dict([(cycle_label, eig.SeriesDimensionReduction(self.raw_data[cycle_label], reload_path=os.path.join(sample_path, "processed_data", cycle_label)))
                             for cycle_label in self.raw_data.keys()])
            
            [dim_red_values.perform_pca(10) for dim_red_values in self.dim_red.values()];
            
            self.peak_selections = {}

            if fit_results==True:
                for peak_file in os.listdir(os.path.join(sample_path, "selected_peaks")):
                    try:
                        cycle_label = peak_file.split("_")[0]
                        if "c" in cycle_label:
                            cycle_type = "charge"
                        else:
                            cycle_type = "discharge"
## Ammended 18.08.2022: changed re.findall("\d", cycle_label)[0] to re.findall("\d+", cycle_label)[0]
                        cycle_number = re.findall("\d+", cycle_label)[0]
                        selection = eig.SelectPeaks(self.raw_data[cycle_type+cycle_number].voltage,
                                                    reload=os.path.join(sample_path, "selected_peaks", peak_file))
                        selection.get_checkbox_values()

                        if cycle_type+cycle_number in self.peak_selections.keys():
                            self.peak_selections[cycle_type+cycle_number].append(selection)
                        else:
                            self.peak_selections.update([(cycle_type+cycle_number, [selection])])
                    except:
                        pass
                
                self.fit_results = dict([(cycle_label,
                     eig.FitResults(self.dim_red[cycle_label], SelectPeaks_objects=self.peak_selections[cycle_label]))
                    for cycle_label in self.peak_selections])
                
        class _KNN_container(object):
            def __init__(container_self):
                pass
        
        self.KNN = _KNN_container()
            
    def two_pass_fit(self, cycle, noise_cutoff_scale=1, center_tolerance=10):
        ## Added center_tolerance as arg 25/10/22
        for voltage in self.dim_red[cycle].voltage[::-1]:
            
            if os.path.isdir(os.path.join(self.sample_path, "Pre-initialised_fit", cycle, "{}V".format(voltage))):
                pass

            else:
                print(voltage)
                x_extent = self.dim_red[cycle].common.x_extent
                y_extent = self.dim_red[cycle].common.y_extent
                grid_reference = np.arange(x_extent*y_extent).reshape(x_extent, y_extent)

                peaks_to_fit = self.fit_results[cycle].peaks_to_fit[voltage]
                unique_peaks = np.unique([item for sublist in [*peaks_to_fit.values()] for item in sublist])
                max_components = max([len(values) for values in self.fit_results[cycle].fit_results.values()])
                reduced_values = self.dim_red[cycle].data_reduced[voltage].reshape((x_extent, y_extent, self.dim_red[cycle].n_components))
                fit_outputs = np.full((x_extent, y_extent), np.nan, dtype=object)
                time_to_fit = np.zeros((x_extent, y_extent))

                for x in notebook(range(x_extent)):
                    for y in notebook(range(y_extent)): ## notebook added for checking, 25/10/22
                        amplitude_scale = np.zeros((unique_peaks.shape[0], max_components))
                        amplitude_value = np.zeros((unique_peaks.shape[0], max_components))
                        for npeak, peak in enumerate(unique_peaks):
                            for ncomp in range(max_components):
                                amplitude_scale[npeak, ncomp] = reduced_values[x, y, ncomp]
                                if peak in self.fit_results[cycle].peaks_to_fit[voltage]["component{}".format(ncomp+1)]:
                                    amplitude_value[npeak, ncomp] = self.fit_results[cycle].fit_results[voltage]["component{}".format(ncomp+1)].best_values["peak_{}_amplitude".format(peak)]

                        scaled_amplitudes = np.sum(amplitude_value*amplitude_scale, axis=1)
                        ## ADDED 25/10/2022
                        norm_spectrum = self.dim_red[cycle].normalised[voltage][grid_reference[x, y]]
                        negative_idx = np.nonzero(norm_spectrum<0)
                        noise_cutoff = -noise_cutoff_scale*np.nanmedian(negative_idx)
                        ### 
                        
                        initial_amplitudes = dict([(peak, scaled_amplitudes[npeak]) for npeak, peak in enumerate(unique_peaks) if scaled_amplitudes[npeak]>noise_cutoff])
                        print(initial_amplitudes)

                        model_xy = LinearModel()
                        for peak in initial_amplitudes.keys():
                            model_xy += LorentzianModel(prefix="peak_{}_".format(peak))
                        params_xy = model_xy.make_params()

                        for name in params_xy:
                            if name not in ["slope", "intercept"]:
                                center = int(name.split("_")[1])
                                if "center" in name:
                                    params_xy[name].value = center
                                    params_xy[name].min = center-center_tolerance
                                    params_xy[name].max = center+center_tolerance
                                if "sigma" in name:
                                    params_xy[name].min = 1
                                    params_xy[name].max = 50
                                    params_xy[name].value = 2
                                if "amplitude" in name:
                                    params_xy[name].min = 0
                                    params_xy[name].value = initial_amplitudes[center]
                        fit_start_time = datetime.now()
                        ## Hack added 27/07/2022 - shift range clipped because of inconsistency in 210616 data
                        spectrum_intensity = self.dim_red[cycle].normalised[voltage].reshape(x_extent, y_extent, self.dim_red[cycle].normalised[voltage].shape[-1])[x, y, :]
                        spectrum_shift = self.dim_red[cycle].common.shift[:spectrum_intensity.shape[0]]
                        fit_outputs[x, y] = model_xy.fit(x=spectrum_shift,
                                                  data=spectrum_intensity,
                                                  params=params_xy)
                        fit_end_time = datetime.now()
                        time_to_fit[x, y] = (fit_end_time-fit_start_time).seconds
                ## Hack - see line 250
                fit_results_spectra = np.zeros((x_extent, y_extent, spectrum_intensity.shape[0]
#                                                 self.dim_red[cycle].common.shift_extent
                                               ))
                fit_results_values = {}

                for x in range(x_extent):
                    for y in range(y_extent):
                        fit_results_spectra[x, y, :] = fit_outputs[x, y].best_fit
                        for keys, values in fit_outputs[x, y].best_values.items():
                            if keys not in fit_results_values.keys():
                                fit_results_values.update([(keys, np.full((x_extent, y_extent), np.nan))])
                            fit_results_values[keys][x, y] = values


                if os.path.isdir(os.path.join(self.sample_path, "Pre-initialised_fit", cycle, "{}V".format(voltage))) == False:
                    os.makedirs(os.path.join(self.sample_path, "Pre-initialised_fit", cycle, "{}V".format(voltage)))

                np.save(os.path.join(self.sample_path, "Pre-initialised_fit", cycle, "{}V".format(voltage), "fit_results_spectra.npy"), fit_results_spectra, allow_pickle=True)
                np.save(os.path.join(self.sample_path, "Pre-initialised_fit", cycle, "{}V".format(voltage), "fit_results_values.npy"), fit_results_values, allow_pickle=True)
                np.save(os.path.join(self.sample_path, "Pre-initialised_fit", cycle, "{}V".format(voltage), "time_to_fit.npy"), time_to_fit, allow_pickle=True)
        
        setattr(self, cycle, _TwoPass(self=self, cycle=cycle))
        

        
    def knn_select(self, cycle, x_component=0, y_component=1, n_neighbours=3, random_seed=0, n_training_spectra=100):
        setattr(self.KNN, cycle, _KNNpeaks(self=self, cycle=cycle, x_component=x_component,
                                           y_component=y_component, random_seed=random_seed, n_training_spectra=n_training_spectra))
        
        