import matplotlib.pyplot as plt
import numpy as np
import os
import EigenFit as eig
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
    
## ADDITIONS JAN 23
def discharge_voltages(cycle, voltage_list, volt_max=2.8, volt_min=1.5):
    voltages = np.sort(voltage_list)[::-1]
    return (2*cycle-1)*(volt_max-volt_min)-(voltages-volt_min)

def charge_voltages(cycle, voltage_list, volt_max=2.8, volt_min=1.5):
    voltages = np.sort(voltage_list)
    return (2*cycle-1)*(volt_max-volt_min)+(voltages-volt_min)
## END OF ADDITIONS
    
class _TwoPass(object):
    def __init__(two_pass_self, self, cycle):
        load_path = os.path.join(self.sample_path, "2_pass_fit", cycle)
        two_pass_self.values = dict([(float(re.findall("\d+.\d+", voltage)[0]),
                                             np.load(os.path.join(load_path, voltage, "fitted_lorentzian_params.npy"),
                                                     allow_pickle=True).item())
                                           for voltage in os.listdir(os.path.join(load_path))])
        two_pass_self.spectra = dict([(float(re.findall("\d+.\d+", voltage)[0]),
                                             np.load(os.path.join(load_path, voltage, "fitted_spectra.npy"),
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
        
    ### Updated 22/01/2023
    def fit_model(knn_self, label_key):
        from sklearn.neighbors import KNeighborsClassifier
        self = knn_self._outer_self
        voltages = self.dim_red[knn_self._cycle].voltage

        if type(label_key) == str:
            label_key = [label_key]

        knn_self.predict_dict = dict([(voltage, np.zeros((knn_self.x_extent, knn_self.y_extent), dtype=int))
                                      for voltage in voltages])

        for label in label_key:
            y_train = np.zeros((knn_self.n_training_spectra), dtype=int)
            for idx in knn_self.label_lists[label]:
                y_train[idx] = 1
            model = KNeighborsClassifier(n_neighbors=knn_self.n_neighbours)

            x_test= [knn_self._ft[idx, [knn_self.x_component, knn_self.y_component]]
                       for idx in range(knn_self._ft.shape[0]) if idx not in knn_self.spectra_list]

            model.fit(knn_self.features, y_train)
            y_predict = model.predict(x_test)

            test_idx = [n for n in np.arange(knn_self.merged.shape[0]) if n not in knn_self.spectra_list]

            merged_predict = np.full((knn_self.merged.shape[0]), np.nan)
            for n, idx in enumerate(knn_self.spectra_list):
                merged_predict[idx] = y_train[n]
            for n, idx in enumerate(test_idx):
                merged_predict[idx] = y_predict[n]

            predict_label = dict([(voltage, 
                                   merged_predict[n*knn_self.data_extent:(n+1)*knn_self.data_extent].reshape(knn_self.x_extent, knn_self.y_extent))
                                 for n, voltage in enumerate(voltages)])

            for voltage in voltages:
                knn_self.predict_dict[voltage][predict_label[voltage]==1] = 1
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
        """
        Methods
        ----------
        two_pass_fit(cycle, center_tolerance=10, sigma_max=50, verbose=True)
        
        one_pass_fit(cycle, peaks_1PF=None, center_tolerance=10, sigma_max=50, verbose=True)
        """
        
        self.sample_path = sample_path
        
        ## Added 21/01/2023 for compatibility with additional plotting functions
        self.peak_dict = {218: "S$_{8}$",
                 202: "S$_{4}^{2-}$",
                 398: "I(398)",##"S$_{6, 8}^{2-}$",
                 450: "I(450)", ##"S$_{4, 2}^{2-}$",
                 504: "S$_{6}^{2-}$"}
        
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
        
   ### CHANGED 12/11/2022
   ### CHANGED 20/11/2022
    
    def two_pass_fit(self, cycle, center_tolerance=10, sigma_max=50, verbose=True):
        x_extent = self.dim_red[cycle].common.x_extent
        y_extent = self.dim_red[cycle].common.y_extent
        shift_extent = self.dim_red[cycle].common.shift_extent
        grid_ref = np.arange(x_extent*y_extent).reshape(x_extent, y_extent)
        
        def _2_pass_pos(x, y, spectrum_fit_arr, values_fit, time_to_fit):
            start_time = datetime.now()
            model = LinearModel()


            neg_idx = np.nonzero(np.sign(self.dim_red[cycle].normalised[voltage][grid_ref[x, y]])==-1)
            negatives = self.dim_red[cycle].normalised[voltage][grid_ref[x, y]][neg_idx]

            all_peaks_to_fit = np.unique([item for sublist in [*self.fit_results[cycle].peaks_to_fit[voltage].values()] for item in sublist])
            positive_peaks = []

            for peak in all_peaks_to_fit:
                if self.fit_results[cycle].scaled_results[peak][voltage][x, y] > -0.5*np.nanmedian(negatives):
                    model += LorentzianModel(prefix="peak_{}_".format(peak))
                    params = model.make_params()
                    positive_peaks.append(peak)

            try:
                for peak in positive_peaks:
                    peak_sigma = np.nanmedian(([self.fit_results[cycle].fit_results[voltage][keys].best_values["peak_{}_sigma".format(peak)] 
                                                for keys, values in self.fit_results[cycle].peaks_to_fit[voltage].items() 
                                                if peak in values]))

                    for name in params:
                        if name not in ["intercept", "slope"]:
                            center = int(name.split("_")[1])
                            if "center" in name and str(peak) in name:
                                params[name].value = center
                                params[name].min = center-center_tolerance
                                params[name].max = center+center_tolerance
                            if "sigma" in name and str(peak) in name:
                                params[name].min = 1
                                params[name].max = sigma_max
                                params[name].value = peak_sigma
                            if "amplitude" in name and str(peak) in name:
                                params[name].value = self.fit_results[cycle].scaled_results[peak][voltage][x, y]*np.pi*peak_sigma   
                                params[name].min = 0

                fit_result = model.fit(x=self.dim_red[cycle].common.shift,
                         data=self.dim_red[cycle].normalised[voltage][grid_ref[x, y]],
                          params=params)
                end_time = datetime.now()

                spectrum_fit_arr[x, y, :] = fit_result.best_fit

                for peak in positive_peaks:
                    if peak not in values_fit.keys():
                        values_fit.update([(peak, {"center": np.full((x_extent, y_extent), np.nan),
                                                   "sigma": np.full((x_extent, y_extent), np.nan),
                                                   "amplitude": np.full((x_extent, y_extent), np.nan)})])
                    for parameter in ["center", "sigma", "amplitude"]:
                        values_fit[peak][parameter][x, y] = fit_result.best_values["peak_{}_{}".format(peak, parameter)]
                time_to_fit[x, y] = (end_time-start_time).microseconds
            except:
                pass
            
        
        for voltage in self.dim_red[cycle].voltage:
            if verbose == True:
                print(voltage)

            save_path_2PF = os.path.join(self.sample_path, "2_pass_fit", cycle, str(voltage))
            if os.path.isdir(save_path_2PF) == False:
                os.makedirs(save_path_2PF)

                spectrum_fit_arr = np.full((x_extent, y_extent, shift_extent), np.nan)
                values_fit = {}
                time_to_fit = np.full((x_extent, y_extent), np.nan, dtype=float)

                if verbose == True:
                    for x in notebook(range(x_extent)):
                        for y in range(y_extent):
                             _2_pass_pos(x, y, spectrum_fit_arr, values_fit, time_to_fit)
                else:
                    for x in range(x_extent):
                        for y in range(y_extent):
                            _2_pass_pos(x, y, spectrum_fit_arr, values_fit, time_to_fit)

                np.save(os.path.join(save_path_2PF, "fitted_spectra.npy"), spectrum_fit_arr, allow_pickle=True)
                np.save(os.path.join(save_path_2PF, "fitted_lorentzian_params.npy"), values_fit, allow_pickle=True)
                np.save(os.path.join(save_path_2PF, "time_to_fit.npy"), time_to_fit, allow_pickle=True)
        
        setattr(self, cycle, _TwoPass(self=self, cycle=cycle))
        
    def one_pass_fit(self, cycle, peaks_1PF=None, center_tolerance=10, sigma_max=50, verbose=True): 
        x_extent = self.dim_red[cycle].common.x_extent
        y_extent = self.dim_red[cycle].common.y_extent
        shift_extent = self.dim_red[cycle].common.shift_extent
        shift = self.dim_red[cycle].common.shift
        grid_ref = np.arange(x_extent*y_extent).reshape(x_extent, y_extent)
        
        
        for voltage in self.dim_red[cycle].voltage[::-1]:
            if verbose == True:
                print(voltage)
                
            if peaks_1PF == None:
                peaks_1PF = [152, 202, 218, 248, 398, 438, 450, 473, 504]
            save_path_1PF = os.path.join(self.sample_path, "1_pass_fit", cycle, str(voltage))
            if os.path.isdir(save_path_1PF) == False:
                os.makedirs(save_path_1PF)

                spectra_1PF = np.full((x_extent, y_extent, shift_extent), np.nan)
                values_1PF = dict([(peak, {"center": np.full((x_extent, y_extent), np.nan),
                                           "sigma": np.full((x_extent, y_extent), np.nan),
                                           "amplitude": np.full((x_extent, y_extent), np.nan)})
                                  for peak in peaks_1PF])
                time_to_fit_1PF = np.full((x_extent, y_extent), np.nan, dtype=float)

                model_1PF = LinearModel()
                for peak in peaks_1PF:
                    model_1PF += LorentzianModel(prefix="peak_{}_".format(peak))

                for x in notebook(range(x_extent)):
                    for y in notebook(range(y_extent)):
                        start_time = datetime.now()

                        params_1PF = model_1PF.make_params()
                        for name in params_1PF:
                            if name not in ["intercept", "slope"]:
                                center = int(name.split("_")[1])
                                if "center" in name:
                                    params_1PF[name].value = center
                                    params_1PF[name].min = center-center_tolerance
                                    params_1PF[name].max = center+center_tolerance
                                if "sigma" in name:
                                    params_1PF[name].min = 1
                                    params_1PF[name].max = sigma_max
                                    params_1PF[name].value = 2
                                if "amplitude" in name:
                                    params_1PF[name].min = 0

                        model_fit_1PF = model_1PF.fit(x=shift, 
                                      data=self.dim_red[cycle].normalised[voltage][grid_ref[x, y]],
                                      params=params_1PF)
                        spectra_1PF[x, y, :] = model_fit_1PF.best_fit
                        for peak in peaks_1PF:
                            for name in ["center", "sigma", "amplitude"]:
                                values_1PF[peak][name][x, y] = model_fit_1PF.best_values["peak_{}_{}".format(peak, name)]

                        end_time = datetime.now()
                        time_to_fit_1PF[x, y] = (end_time-start_time).microseconds

                np.save(os.path.join(save_path_1PF, "fitted_spectra.npy"), spectra_1PF, allow_pickle=True)
                np.save(os.path.join(save_path_1PF, "fitted_lorentzian_params.npy"), values_1PF, allow_pickle=True)
                np.save(os.path.join(save_path_1PF, "time_to_fit.npy"), time_to_fit_1PF, allow_pickle=True)
        
    def knn_select(self, cycle, x_component=0, y_component=1, n_neighbours=3, random_seed=0, n_training_spectra=100):
        setattr(self.KNN, cycle, _KNNpeaks(self=self, cycle=cycle, x_component=x_component,
                                           y_component=y_component, random_seed=random_seed, n_training_spectra=n_training_spectra))
        
        
#### ADDITIONS JAN 2023
    def make_intensities(self, cycle):
        all_peaks = np.unique(np.hstack([[*voltage_values.keys()] for voltage_values in vars(self)[cycle].values.values()]))
        x_extent = self.raw_data[cycle].common.x_extent
        y_extent = self.raw_data[cycle].common.y_extent

        intensities = {}
        for voltage, voltage_values in vars(self)[cycle].values.items():
            intensities.update([(voltage, {})])
            for peak in all_peaks:
                if peak in voltage_values.keys():
                    intensities[voltage].update([(peak,
                                                  voltage_values[peak]["amplitude"]/voltage_values[peak]["sigma"]/np.pi)])
                else:
                    intensities[voltage].update([(peak, np.full((x_extent, y_extent), np.nan))])

        return intensities

    def make_bplot_intensities(self, cycle, peak, predict_dict, discharge=True):
        intensity_data = self.make_intensities(cycle)
               
        if discharge==True:
            voltages = np.sort([*intensity_data.keys()])[::-1]
        else:
            voltages = np.sort([*intensity_data.keys()])

        bp_i = dict([(peak, dict([(voltage, np.array([])) 
                                  for voltage in voltages]))])

        for voltage in voltages:
            if peak in np.sort([*intensity_data[voltage].keys()])[::-1]:
                try:
                    bp_i[peak][voltage] = intensity_data[voltage][peak][predict_dict[voltage]==1][
                                np.isfinite(intensity_data[voltage][peak][predict_dict[voltage]==1])
                    ]
                except:
                    bp_i[peak][voltage] = np.array([])
        return bp_i
    
    def PS_peak_scatter(self, cycle, labels_to_fit, discharge=True, peak_plots=[398, 450], voltage_cutoff=2.1):
        all_peaks = np.unique(np.hstack([[*voltage_values.keys()] for voltage_values in vars(self)[cycle].values.values()]))
        x_extent = self.raw_data[cycle].common.x_extent
        y_extent = self.raw_data[cycle].common.y_extent

        intensities = self.make_intensities(cycle)
        
        vars(self.KNN)[cycle].fit_model(labels_to_fit)
        predict_dict = vars(self.KNN)[cycle].predict_dict

        intensity_filtered = dict([(voltage, intensities[voltage][peak_plots[0]][predict_dict[voltage]==1])
                                    for voltage in intensities if peak_plots[0] in intensities[voltage].keys()
                                    and intensities[voltage][peak_plots[0]][predict_dict[voltage]==1].shape[0]>0])

        voltage = intensity_filtered.keys()
        if voltage_cutoff != None:
            lower_peak = [v for v in voltage if v<voltage_cutoff]
            upper_peak = [v for v in voltage if v>=voltage_cutoff]

            upper_colors = [plt.cm.Reds_r(i) for i in np.linspace(0.2, 0.8, len(upper_peak))]
            lower_colors = [plt.cm.Blues_r(i) for i in np.linspace(0.2, 0.8, len(lower_peak))]

            colors = upper_colors+lower_colors
        else:
            colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(voltage))]

        markers = ["o", "s", "^", "v"]

        vmax = eig.percentile_vmax(np.hstack((values[peak].flatten() for values in intensities.values()
                                              for peak in values.keys() if peak in peak_plots)))

        f, ax = plt.subplots()

        ax.set_prop_cycle("marker", markers)

        for nvolt, voltage in enumerate([*intensity_filtered.keys()][::-1]):
            ax.plot(intensities[voltage][peak_plots[0]][predict_dict[voltage]==1],
                        intensities[voltage][peak_plots[1]][predict_dict[voltage]==1],
                        label=voltage, color=colors[nvolt], ls="none")

        ax.set_xlabel("{} (a.u.)".format(self.peak_dict[peak_plots[0]]))
        ax.set_ylabel("{} (a.u.)".format(self.peak_dict[peak_plots[1]]))

        x_min = ax.get_xlim()
        y_min = ax.get_ylim()
        ax.plot([x_min[0], x_min[1]], [y_min[0], y_min[1]],
                color="k", zorder=0, marker=None, lw=0.5)
        ax.set_xlim(x_min)
        ax.set_ylim(y_min)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_aspect("equal")

        return f, ax        
    
    def heatmaps(self, cycle, discharge=True, peaks_to_plot=[218, 202, 398, 450, 504], voltages_to_plot="all", titles=None):
        
        if titles == None:
            titles = ["I({})".format(peak) for peak in peaks_to_plot]
            

        
        all_peaks = np.unique(np.hstack([[*voltage_values.keys()] for voltage_values in vars(self)[cycle].values.values()]))
        x_extent = self.raw_data[cycle].common.x_extent
        y_extent = self.raw_data[cycle].common.y_extent
        intensities = self.make_intensities(cycle)
        
        if voltages_to_plot=="all":
            voltages = [*intensities.keys()]
        else:
            voltages = voltages_to_plot
            
        if discharge == True:
            voltages = np.sort(list(voltages))[::-1]
        else:
            voltages = np.sort(list(voltages))
                
                
        vmax = eig.percentile_vmax(np.hstack((values[peak].flatten() for values in intensities.values()
                                              for peak in values.keys() if peak in peaks_to_plot)))

        f, (axes) = plt.subplots(len(voltages), len(peaks_to_plot),
                                 figsize=(len(peaks_to_plot), len(voltages)))
        for npeak, peak in enumerate(peaks_to_plot):
            axes[0, npeak].set_title(titles[npeak])
            for nvolt, volt in enumerate(voltages):
                axes[nvolt, 0].set_ylabel("{} V".format(volt))
                axes[nvolt, npeak].set_xticks([])
                axes[nvolt, npeak].set_yticks([])
                try:
                    axes[nvolt, npeak].imshow(intensities[volt][peak], vmin=0, vmax=vmax, cmap="YlOrRd")
                except:
                    axes[nvolt, npeak].imshow(np.full((x_extent, y_extent), np.nan))   

        return f, axes    

    
