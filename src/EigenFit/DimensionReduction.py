## VERSION 2
## Updated June 2023

import numpy as np
import os
import matplotlib.pyplot as plt

def snv(input_data):
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])

    return output_data

class DimensionReduction(object):
    def __init__(self, SpectrumMap_obj, npolybackground=4, reload=False, verbose=True, usemethod="IModPoly"):
        from tqdm import notebook
        from BaselineRemoval import BaselineRemoval
        self.x_extent = SpectrumMap_obj.x_extent
        self.y_extent = SpectrumMap_obj.y_extent
        
        def _remove_background():
            if verbose == True:
                if usemethod == "IModPoly":
                    self.background_removed = np.array([BaselineRemoval(spectrum).IModPoly(npolybackground)
                                                    for spectrum in notebook.tqdm(map_reshaped)])
                elif usemethod =="ModPoly":
                    self.background_removed = np.array([BaselineRemoval(spectrum).ModPoly(npolybackground)
                                                    for spectrum in notebook.tqdm(map_reshaped)])
            else:
                if usemethod == "IModPoly":
                    self.background_removed = np.array([BaselineRemoval(spectrum).IModPoly(npolybackground)
                                                    for spectrum in map_reshaped])
                elif usemethod == "ModPoly":
                    self.background_removed = np.array([BaselineRemoval(spectrum).ModPoly(npolybackground)
                                                    for spectrum in map_reshaped])
        
        if "clipped" in vars(SpectrumMap_obj):
            self.shift_extent = SpectrumMap_obj.clipped.shift_extent
            self.shift = SpectrumMap_obj.clipped.shift
            
            map_reshaped = SpectrumMap_obj.clipped.map.reshape((self.x_extent*self.y_extent, self.shift_extent))
        else:
            self.shift_extent = SpectrumMap_obj.raw.shift_extent
            self.shift = SpectrumMap_obj.raw.shift
            
            map_reshaped = SpectrumMap_obj.raw.map.reshape((self.x_extent*self.y_extent, self.shift_extent)) 
  
        if reload == False:
            _remove_background()

        else:
            if os.path.isdir(reload) == False:
                os.makedirs(reload)
            if "background_removed_{}.npy".format(npolybackground) in os.listdir(reload):
                self.background_removed = np.load(os.path.join(reload, "background_removed_{}.npy".format(npolybackground)),
                                                  allow_pickle=True)
            else:
                _remove_background()
                np.save(os.path.join(reload, "background_removed_{}.npy".format(npolybackground)),
                        self.background_removed,
                        allow_pickle=True) 
                
        self.normalised = snv(self.background_removed)
        
        self._background_removed_map = self.background_removed.reshape((self.x_extent, self.y_extent, self.shift_extent))
        self._normalised_map = self.normalised.reshape((self.x_extent, self.y_extent, self.shift_extent))
            
    def apply_pca(self, n_components):
        from sklearn.decomposition import PCA
        
        class _PCA_container(object):
            def __init__(_self):
                pass
            
        self.pca_outputs = _PCA_container()
        
        self.pca_outputs.n_components = n_components
        self.pca_outputs.pca = PCA(n_components=self.pca_outputs.n_components)
        self.pca_outputs.loadings = self.pca_outputs.pca.fit_transform(self.normalised)
        self.pca_outputs.components = self.pca_outputs.pca.components_
        self.pca_outputs.scores = np.dot(self.normalised, self.pca_outputs.components.T)
        self.pca_outputs.reconstructed = np.dot(self.pca_outputs.scores, self.pca_outputs.components)
        
        self.pca_outputs.map_scores = dict([(n, self.pca_outputs.scores[:, n].reshape((self.x_extent, self.y_extent)))
                                            for n in range(self.pca_outputs.n_components)])
        self.pca_outputs.map_reconstructed = self.pca_outputs.reconstructed.reshape((self.x_extent, self.y_extent, self.shift_extent))
        
        
class SeriesDimensionReduction(object):
    def __init__(self, MapSeries_obj, reload=False, npolybackground=4, verbose=True):
        
        from tqdm import notebook
        from BaselineRemoval import BaselineRemoval
        
        self.common = MapSeries_obj.common
        self._MapSeries = MapSeries_obj
        self._series_label = MapSeries_obj._series_by
        setattr(self, self._series_label, 
                vars(MapSeries_obj)[MapSeries_obj._series_by]) ## e.g. self.voltage
        
        if "clipped" in vars([*MapSeries_obj.data.values()][0]).keys():
            map_type = "clipped"
        else:
            map_type = "raw"
            
        target_fname = "series_background_removed_{}_{}_{}.npy".format(npolybackground, np.min(self.common.shift), np.max(self.common.shift))
        
        if reload != False:
            if target_fname not in os.listdir(reload):
                self.background_removed = {}

                for label in notebook.tqdm(vars(self)[self._series_label]):
                    self._map_reshaped = vars(self._MapSeries.data[label])[map_type].map.reshape((self.common.x_extent*self.common.y_extent,
                                                                                            self.common.shift_extent))
                    self.background_removed.update([(label, 
                                                     np.array([BaselineRemoval(spectrum).IModPoly(npolybackground) 
                                                               for spectrum in self._map_reshaped]))])
                
                np.save(os.path.join(reload, target_fname),
                        self.background_removed,
                        allow_pickle=True)
                
            elif target_fname in os.listdir(reload):
                self.background_removed = np.load(os.path.join(reload, target_fname),
                                                  allow_pickle=True).item()
                for label in vars(self)[self._series_label]:
                    if label not in self.background_removed.keys():
                        self.background_removed.update([(label, 
                                                     np.array([BaselineRemoval(spectrum).IModPoly(npolybackground) 
                                                               for spectrum in self._map_reshaped]))])
                
                np.save(os.path.join(reload, target_fname),
                        self.background_removed,
                        allow_pickle=True)
                
        else:
            for label in notebook.tqdm(vars(self)[self._series_label]):
                self._map_reshaped = vars(self._MapSeries.data[label])[map_type].map.reshape((self.common.x_extent*self.common.y_extent,
                                                                                        self.common.shift_extent))
                self.background_removed.update([(label, 
                                                 np.array([BaselineRemoval(spectrum).IModPoly(npolybackground) 
                                                               for spectrum in self._map_reshaped]))])
                
        self._combined_data = snv(np.vstack([*self.background_removed.values()]))
        data_extent = self.common.x_extent*self.common.y_extent
        
        self.normalised = dict([(label, self._combined_data[nlabel*data_extent:(nlabel+1)*data_extent, :])
                                for nlabel, label in enumerate(vars(self)[self._series_label])]) 
        
        self.map_background_removed = dict([(keys, values.reshape((self.common.x_extent, self.common.y_extent, self.common.shift_extent))) for keys, values in self.background_removed.items()])
        self.map_normalised = dict([(keys, values.reshape((self.common.x_extent, self.common.y_extent, self.common.shift_extent))) for keys, values in self.normalised.items()])
        
    def apply_pca(self, n_components=None):
        from sklearn.decomposition import PCA
        
        class _PCA_container(object):
            def __init__(_self):
                pass
            
        setattr(self, "pca_outputs", _PCA_container())
        self.pca_outputs.n_components = n_components
        self.pca_outputs.pca = dict([(label, PCA(self.pca_outputs.n_components)) for label in vars(self)[self._series_label]])
        self.pca_outputs.loadings = dict([(label, values.fit_transform(self.normalised[label]))
                                          for label, values in self.pca_outputs.pca.items()])
        self.pca_outputs.components = dict([(label, values.components_) for label, values in self.pca_outputs.pca.items()])
        self.pca_outputs.scores = dict([(label, np.dot(self.normalised[label], self.pca_outputs.components[label].T))
                                        for label in vars(self)[self._series_label]])
        self.pca_outputs.reconstructed = dict([(label, np.dot(self.pca_outputs.scores[label],
                                                              self.pca_outputs.components[label]))
                                              for label in vars(self)[self._series_label]])
        
        self.pca_outputs.map_scores = dict([(label, values.reshape(self.common.x_extent, self.common.y_extent, self.pca_outputs.n_components))
                                            for label, values in self.pca_outputs.scores.items()])
        self.pca_outputs.map_reconstructed = dict([(label, values.reshape(self.common.x_extent, self.common.y_extent, self.common.shift_extent))
                                            for label, values in self.pca_outputs.reconstructed.items()])
        
    def plot_components(self, components_to_show=3, index=None, offset=0.2):
    
        if index != None:
            idx_to_show = vars(self)[self._series_label][index]
            
            f, ax = plt.subplots()
            [ax.plot(self.common.shift,
                     self.pca_outputs.components[idx_to_show][c_idx, :]-n*offset)
             for n, c_idx in enumerate(range(components_to_show))]
            
        else:
            from ipywidgets import interact, IntSlider, FloatSlider
            f, ax = plt.subplots()
            def update(idx, offset):
                ax.cla()
                label = vars(self)[self._series_label][idx]
                ax.set_title(label)
                [ax.plot(self.common.shift,
                         self.pca_outputs.components[label][c_idx, :]+(components_to_show-i)*offset)
                 for i, c_idx in enumerate(range(components_to_show))];
                ax.set_yticks([])
                [ax.text(x=np.max(self.common.shift), y=(components_to_show-i)*offset, s="PC {}".format(i+1), ha="right") for i in range(components_to_show)];
                [ax.axhline((components_to_show-i)*offset, color="k", lw=0.5) for i in range(components_to_show)];
            interact(update, idx=IntSlider(min=0, max=len(vars(self)[self._series_label])-1),
                     offset=FloatSlider(min=0, max=1, step=0.05, value=0.3))
            
    def get_selected_peaks(self, peak_select_path, label):
        import glob
        import os
        from SelectPeaks import SelectPeaks
        
        self.selected_peaks = [SelectPeaks(voltage_series=vars(self)[self._series_label], reload=fname)
                      for fname in glob.glob(os.path.join(peak_select_path, "*.npy")) if label in fname]
        [sp.get_checkbox_values() for sp in self.selected_peaks];
