import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider

try:
    from tqdm import notebook
    from notebook import tqdm as notebook
except:
    from tqdm import tqdm_notebook as notebook
    
from BaselineRemoval import BaselineRemoval
from snv import snv

from sklearn.decomposition import PCA

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

class DimensionReduction(object):
    def __init__(self, SpectrumMap_object): 
        """
        Removes the background and normalises SpectrumMap objects
        """
        try:
            map_reshaped = SpectrumMap_object.clipped.map.reshape(SpectrumMap_object.x_extent*SpectrumMap_object.y_extent,
                                                   SpectrumMap_object.clipped.shift_extent)
            self.shift = SpectrumMap_object.clipped.shift
        except:
            map_reshaped = SpectrumMap_object.raw.map.reshape(SpectrumMap_object.x_extent*SpectrumMap_object.y_extent,
                                               SpectrumMap_object.raw.shift_extent)
            self.shift = SpectrumMap_object.raw.shift
            
        self.background_removed = np.array([BaselineRemoval(spectrum).IModPoly(4) for spectrum in notebook(map_reshaped)])
        self.normalised = snv(self.background_removed)
        
    def perform_pca(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=None)
        self.fit_transform = self.pca.fit_transform(self.normalised)
        self.components = self.pca.components_
        self.data_reduced = np.dot(self.normalised,
                                   self.components[:n_components, :].T)
        self.data_reconstructed = np.dot(self.data_reduced,
                                        self.components[:n_components, :])
    def visualise_components(self, components_to_display=3):
        f, ax = plt.subplots()
        [ax.plot(self.shift, self.components[i]+0.3*(components_to_display-i)) for i in range(components_to_display)];
        [ax.axhline(0.3*(components_to_display-i), color="k", lw=0.5) for i in range(components_to_display)];
        return f, ax
        

class SeriesDimensionReduction(object):
    def __init__(self, MapSeries_object, reload_path="none", background_degree=4):
        self.common = MapSeries_object.common
        self._MapSeries = MapSeries_object
        self._series_label = MapSeries_object._series_by
        setattr(self, self._series_label, vars(MapSeries_object)[MapSeries_object._series_by])
        
        if reload_path == "none" or "background_removed_degree_{}.npy".format(background_degree) not in os.listdir(reload_path):      
            map_reshaped = {}
            self.background_removed = {}
            for idx in notebook(vars(self)[self._series_label]):    ### equivalent to each voltage
                
                try:
                    map_reshaped.update([(idx, 
                                          MapSeries_object.data[idx].clipped.map.reshape(self.common.x_extent*self.common.y_extent, 
                                                                                    MapSeries_object.data[idx].clipped.shift_extent))])
                except:
                    map_reshaped.update([(idx, MapSeries_object.data[idx].raw.map.reshape(self.common.x_extent*self.common.y_extent, 
                                                                              MapSeries_object.data[idx].raw.shift_extent))])                 
                
                self.background_removed.update([(idx,
                                                  np.array([BaselineRemoval(spectrum).IModPoly(background_degree) for\
                                                            spectrum in map_reshaped[idx]]))
                                               ])
                
                if reload_path != "none":
                    np.save(os.path.join(reload_path, "background_removed_degree_{}.npy".format(background_degree)),
                            self.background_removed,
                           allow_pickle=True)
        else:
            map_reshaped = {}

            self.background_removed = np.load(os.path.join(reload_path, "background_removed_degree_{}.npy".format(background_degree)),
                            allow_pickle=True).item()
            for idx in notebook(vars(self)[self._series_label]):    ### equivalent to each voltage
                if idx not in self.background_removed.keys():
                    try:
                        map_reshaped.update([(idx, 
                                              MapSeries_object.data[idx].clipped.map.reshape(self.common.x_extent*self.common.y_extent, 
                                                                                        MapSeries_object.data[idx].clipped.shift_extent))])
                    except:
                        map_reshaped.update([(idx, MapSeries_object.data[idx].raw.map.reshape(self.common.x_extent*self.common.y_extent, 
                                                                                  MapSeries_object.data[idx].raw.shift_extent))])                 

                    self.background_removed.update([(idx,
                                                      np.array([BaselineRemoval(spectrum).IModPoly(background_degree) for\
                                                                spectrum in map_reshaped[idx]]))
                                                   ])       
            np.save(os.path.join(reload_path, "background_removed_degree_{}.npy".format(background_degree)),
                                self.background_removed,
                               allow_pickle=True)
        
        min_shift_length = min([[*self.background_removed.values()][i].shape[1] for i in range(len(self.background_removed))])
        
        combined_data = np.vstack((np.array([*self.background_removed.values()][i][:, :min_shift_length])) for i in range(len(self.background_removed)))
        normalised_data = snv(combined_data)
        
        self.normalised = dict([(idx,
                                 normalised_data[n_idx*(self.common.x_extent*self.common.y_extent): 
                                                 (n_idx+1)*(self.common.x_extent*self.common.y_extent)])
                               for n_idx, idx in enumerate(vars(self)[self._series_label])
                               ])
        
    def perform_pca(self, n_components=None):
        self.pca = dict([(idx, PCA(n_components=None)) for idx in (vars(self)[self._series_label])]) ## n_components used defined manually later
        self.n_components = n_components
        self.fit_transformed = dict([(idx, self.pca[idx].fit_transform(self.normalised[idx]))
                                     for idx in self.pca.keys()])
        
        self.components = dict([(idx, self.pca[idx].components_) for idx in self.pca.keys()])
        self.inverse_tranformed = dict([(idx, 
                                         self.pca[idx].inverse_transform(self.fit_transformed[idx])) for idx in self.pca.keys()])
        self.data_reduced = dict([(idx,
                                   np.dot(self.normalised[idx],
                                          self.components[idx][:n_components, :].T))
                                  for idx in self.pca.keys()])
        self.data_reconstructed = dict([(idx,
                                         np.dot(self.data_reduced[idx], self.components[idx][:n_components, :]))
                                        for idx in self.pca.keys()])
    
    def visualise_components(self, components_to_display=3):   
        f, ax = plt.subplots()
    
        def update(v_idx, offset):
            ax.cla()
            voltage = [*(vars(self)[self._series_label])][v_idx]
            ax.set_title(" {} V".format(voltage))
            ax.tick_params(which="both", tickdir="in", top=True, right=True)
            ax.set_yticks([])
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            [ax.plot(self.common.shift, self.components[voltage][i, :]+(components_to_display-i)*offset) for i in range(components_to_display)];
            [ax.text(x=np.max(self.common.shift), y=(components_to_display-i)*offset, s="PC {}".format(i+1), ha="right") for i in range(components_to_display)];
            [ax.axhline((components_to_display-i)*offset, color="k", lw=0.5) for i in range(components_to_display)];
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.xaxis.set_minor_locator(MultipleLocator(20))
        interact(update, v_idx=IntSlider(min=0, max=len((vars(self)[self._series_label]))-1, step=1),
                offset=FloatSlider(min=0, max=1, step=0.05, value=0.3))
        return f, ax
