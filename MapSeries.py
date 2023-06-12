### Previously named LoadReload
## Moved to EigenFit 06/01/2022 and combined with _SimpleLoad
## 06/01/2022:
### Generalised to get_sequence_id and MapSeries, with changeable argument to search for (sequence label) rather than always specified as "cellvoltage"
## 02/02/2022
### Removed tqdm loadbar from verbose=False in update method
### Set "voltage" as default keyword for series_by in _SimpleLoad and MapSeries
### Added x_coords and y_coords to _Common and _find_common

## 05/02/2022
## Added quick_stack_view method

## 03/03/2022
## Added cosmic_ray_view method
## Added docstrings

import os
import numpy as np

import re
import glob
try:
    from tqdm import notebook
    from notebook import tqdm as notebook
except:
    from tqdm import tqdm_notebook as notebook
    
from SpectrumMap import SpectrumMap

def get_sequence_id(fname, sequence_label="cellvoltage"):
    import re
    return [float(re.findall("\d+.\d+", seg)[0]) for seg in os.path.split(fname)[-1].split("_") if sequence_label in seg][0]

class _Common(object):
    def __init__(self):
        attrs = ["x_extent", "y_extent", "shift", "shift_extent", "x_coords", "y_coords"]
        self.__dict__.update([(key, None) for key in attrs])

class _SimpleLoad(object):
    def __init__(self, path, verbose=False, sequence_label="cellvoltage", fname_delimiter="_", series_by="voltage", sort=True):
        """
        path = directory containing data files for sequence of measurements
        verbose = show tqdm loading bar if True
        sequence_label = string within filename indicating wich substring to use to label items in sequence
        fname_delimiter = filenames with multiple substrings can be split by delimiter
        series_by = e.g. time series, voltage series. The dimension in which the measurement series changes.
        """
        self._path = path
        self._sort = sort
        self.common = _Common()
        
        if verbose == True:
            self.data = dict([(get_sequence_id(fname, sequence_label), SpectrumMap(fname))\
                               for fname in notebook(glob.glob(os.path.join(path, "*.txt")))
                              ])
        elif verbose == False:
            self.data = dict([(get_sequence_id(fname, sequence_label), SpectrumMap(fname))\
                               for fname in glob.glob(os.path.join(path, "*.txt"))
                              ])
        if series_by is None:
            self._series_by = "series"
        else:
            self._series_by = series_by
        self._sequence_label= sequence_label
        setattr(self, self._series_by, [*self.data.keys()])
        self._find_common()
        
    def update(self, verbose=False):
        """
        Updates the MapSeries with any new data in the load_path directory.
        Useful if updating MapSeries on the fly (while adding new .txt files to the directory) and re-sorts series labels from lowest to highest if sort=True
        Inputs
        ---------
        verbose (bool) default=False:
            displays a loading bar if True
        """
        previous_keys = [*self.data.keys()]
        if verbose == True:
            self.data.update([(get_sequence_id(fname, self._sequence_label), SpectrumMap(fname))\
                               for fname in notebook(glob.glob(os.path.join(self._path, "*.txt")))
                              if get_sequence_id(fname, self._sequence_label) not in self.data.keys()])
        elif verbose == False:
            self.data.update([(get_sequence_id(fname, self._sequence_label), SpectrumMap(fname))\
                               for fname in glob.glob(os.path.join(self._path, "*.txt"))
                              if get_sequence_id(fname, self._sequence_label) not in self.data.keys()])            
        
        if self._sort == True:
            self.data = dict([(key, self.data[key]) for key in np.sort([*self.data.keys()])])
        vars(self)[self._series_by] = [*self.data.keys()]
         
        updated_keys = [name for name in [*self.data.keys()] if name not in previous_keys]
        if verbose == True:
            print("Previous keys: \n {}".format(previous_keys))
            print("New keys: \n {}".format(updated_keys))   
        self._find_common()
        
    def _find_common(self, clipped=False):
        """
        hidden utility for updating the common values
        """
        if "common" not in vars(self):
            self.common = _Common()
            
        for key in ["x_coords", "y_coords"]:
            vars(self.common)[key] = vars([*self.data.values()][0])[key]

        for key in ["x_extent", "y_extent"]:      
            if np.unique([vars(meas)[key] for meas in self.data.values()]).shape[0] == 1:
                vars(self.common)[key] = vars([*self.data.values()][0])[key]
            else:
                vars(self.common)[key] = None
                
        for key in ["shift", "shift_extent"]:
            if clipped==True:
                vars(self.common)[key] = vars([*self.data.values()][0].clipped)[key]
            else:
                vars(self.common)[key] = vars([*self.data.values()][0].raw)[key]
        
    def set_clip_range(self, start_shift=None, end_shift=None):
        """
        select a sub-set of data within a range of shift values for further analyis (applied to all map data in series simultaneously)
        
        Inputs
        ----------
        start_shift (float) default=None:
            Lowest shift value to include in the sub-set (uses first measured value if None)
        end_shift (float) default=None:
            Highest shift value to include in the sub-set (uses last measured value if None)
        """
        for key, measurement in self.data.items():
            measurement.set_clip_range(start_shift, end_shift)
 
## Move process to dimension reduction 
#         clipped_ranges = [measurement.clipped.map.shape[-1] for measurement in self.data.values()]
#         for key, measurement in self.data.items():
#             measurement.clipped.map = measurement.clipped.map[:, :, :np.nanmin(clipped_ranges)]
#         print("using correct code")
            
        self._find_common(clipped=True)  
        
        
    def quick_stack_view(self, offset_max=100):
        """
        view a graph of the spectra, labelled by series value, with interactive sliders to select the (x, y) map position to show
        """
        
        import matplotlib.pyplot as plt
        import numpy as np
        from ipywidgets import interact, IntSlider, FloatSlider
        f, ax = plt.subplots()
        def update(x, y, offset):
            ax.cla()
            ax.set_prop_cycle("color", [plt.cm.viridis(i) for i in np.linspace(0, 1, len(self.voltage))])

            try:
                [ax.plot(self.common.shift, 
                         self.data[volt].clipped.map[x, y, :]+offset*(len(self.voltage)-nvolt),
                         label=volt) for nvolt, volt in enumerate(self.voltage)];
            except:
                [ax.plot(self.common.shift, 
                         self.data[volt].raw.map[x, y, :]+offset*(len(self.voltage)-nvolt),
                         label=volt) for nvolt, volt in enumerate(self.voltage)];

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.tick_params(which="both", tickdir="in", right=True, top=True)
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (a.u.)")
            plt.tight_layout()
            ax.set_position((0.15, 0.15, 0.7, 0.7))
        interact(update, x=IntSlider(min=0, max=self.common.x_extent-1, step=1), y=IntSlider(min=0, max=self.common.y_extent-1),
                offset=IntSlider(min=0, max=offset_max, value=40, step=10))

        return f, ax
    
    def cosmic_ray_view(self, positions_to_view=3,
                        show_all_voltages=True, voltages_to_show=None, first_n=False, last_n=False, n_voltages=2,
                        colors=8, mark_interval=4,
                        figsize=(6, 6)
                       ):
        """
        View a 'stack' of graphs with data for some/ all of the series values with interactive sliders for quick manual identification of cosmic ray spikes. 
        Inputs
        ----------
        positions_to_view (int) default=3:
            the number of axes to show in the vertical stack of plots
        
        show_all_voltages (bool) default=True:
            if True, displays all of the series spectra for each position
            if False, set first_n or last_n to True (see below) or specify voltages_to_show as list
            
        voltages_to_show (list) default=None:
            if show_all_voltages is False, specify a list of series values to show        
        
        first_n (bool) default=False:
            if True, displays the first n_voltages (show_all_voltages=False)
        
        last_n (bool) default=False:
            if True, displays the first n_voltages (show_all_voltages=False)
        
        n_voltages (int) default=2:
            the number of voltages (first_n or last_n) to display
        
        colors (int) default=8:
            the number of different colors to use to display voltages (NB plots also differentiated by markers)
        
        mark_interval (int) default=4:
            interval between markers. Set to 1 to mark every point (takes longer for graph to load)
        
        figsize (tuple) default=(6, 6):
            size of the matplotlib figure
        """
        
        import matplotlib.pyplot as plt
        import numpy as np
        from ipywidgets import interact, IntSlider, FloatSlider
        n_segments = int(np.ceil(self.common.x_extent/positions_to_view))

        if first_n == True:
            voltages_to_plot = self.voltage[:n_voltages]
        if last_n == True:
            voltages_to_plot = self.voltage[-n_voltages:]
        elif show_all_voltages == True:
            voltages_to_plot = self.voltage

        ax_sort = dict([(n, 
                     np.arange(self.common.x_extent)[positions_to_view*n:positions_to_view*(n+1)]) 
                        for n in range(n_segments)])

        if "clipped" in vars([*self.data.values()][0]):
            disp = "clipped"
        else:
            disp = "raw"

        markers = ["o", "s", "d", "^", "v"]
        colors = [plt.cm.tab10(i) for i in range(colors)]

        color_id = 0
        marker_id = 0
        plot_colors = []
        plot_markers = []

        for nv, v in enumerate(voltages_to_plot):
            if color_id < len(colors):
                plot_colors.append(colors[color_id])
                plot_markers.append(markers[marker_id])
                color_id += 1
            elif color_id == len(colors):   
                color_id = 0
                marker_id += 1
                plot_colors.append(colors[color_id])
                plot_markers.append(markers[marker_id])  

        f, (axes) = plt.subplots(positions_to_view, 1, figsize=(figsize))

        for axis in axes:
            pos = axis.get_position()
            axis.set_position((pos.x0, pos.y0+0.05, 0.8*pos.width, pos.height))

        def update(seg_id, x):
            for row in range(positions_to_view):
                axes[row].cla()
            for row in range(positions_to_view):
                try:
                    axes[row].set_ylabel("x={}, y={}".format(x, ax_sort[seg_id][row]))
                    [axes[row].plot(self.common.shift, 
                                    vars(self.data[volt])[disp].map[x, ax_sort[seg_id][row], :],
                                    marker=plot_markers[nvolt],
                                    markevery=4,
                                    label=volt) for nvolt, volt in enumerate(voltages_to_plot)];
                except:
                    pass

            handles, labels = axes[0].get_legend_handles_labels()
            f.legend(handles, labels, loc="center left", bbox_to_anchor=(0.8, 0.5))
        interact(update, seg_id=IntSlider(min=0, max=n_segments-1, step=1),
                         x=IntSlider(min=0, max=self.common.x_extent-1, step=1))

        return f, (axes)
    
## Added 04/06/2023
    def apply_CRE(self, CRE_path):
        cre = np.load(CRE_path, allow_pickle=True).item()
        self.cosmic_rays = cre
        for nvolt, volt in enumerate(cre["voltage"]):
            try:
                self.data[volt].fix_cosmic_ray(x_pos=cre["x_pos"][nvolt], y_pos=cre["y_pos"][nvolt], 
                                       spike_pos=cre["spike_pos"][nvolt], smooth_width=cre["smooth_width"][nvolt])
            except:
                print(volt)
    

def MapSeries(load_path, verbose=False,
              sequence_label="cellvoltage", fname_delimiter="_",
              series_by="voltage", sort=True):
    """
    Creates, loads, or updates a series of Raman maps sorted by a series label, and includes functions for pre-processing and viewing the raw data prior to fitting.
    
    Inputs
    ----------
    load_path (str):
        path to a directory containing text files to load into a map series (possibly with pre-created VoltageSeries.npy file)
    verbose (bool) default = False:
        displays a loading bar if True
    sequence_label (str) default = "cellvoltage":
        the label used in filenames to indicate the substring containing the series value
    fname_delimiter (str) default = "_"
        character used to separate filename into substrings to find label
    series_by (str) default = "voltage"
        the label given to the list of series values in the MapSeries object
    sort (bool) default = True
        sort the data series from lowest to highest
        
    Methods
    ----------
    update:
        appends new data files
    set_clip_range:
        select a sub-set of the shift axis values
    quick_stack_view:
        displays all the spectra measured at the same position for the full series
    cosmic_ray_view:
        displays multiple spectra at a given position with multiple plots for quick identification of cosmic rays
    
    
    Before you start: 
    ----------
    Save the raw Raman data for a series (e.g. time series, voltage series) into a single directory. The filename can be anything, but should include a 'sequence label' and corresponding value delimited by a recognisable character (typically '_')
    
    Example filenames: 'myRamanMap_cellvoltage2.45_514nm_center750.txt' -- sequence_label="cellvoltage", fname_delimiter="_"
                       'myRamanMap-timestamp60secs-514nm.txt" -- sequence_label="timestamp", fname_delimiter="-"
                       
   How to call:
   ----------
   series = MapSeries(load_path=os.path.join("Top_directory", "Series_directory"))
   
   Notes:
   ----------                       
   Creating a MapSeries automatically saves the corresponding formatted data in a .npy file in the load_path directory. 
   New data can be incorporated into a MapSeries object by simply calling the MapSeries function with the load_path specified: this will append new data to the existing .npy file. (This saves time compared to reloading all of the data fresh every time the function is called). 
   Returns _SimpleLoad class (with methods described in this docstring)
                       
    """  
    if "VoltageSeries.npy" in os.listdir(load_path):
        if verbose == True:
            print("Reloading previous SimpleLoad object")
            
        reload = np.load(os.path.join(load_path, "VoltageSeries.npy"), allow_pickle=True).item()
        reload.update(verbose=verbose)
        np.save(os.path.join(load_path, "VoltageSeries.npy"), reload, allow_pickle=True)        
        return reload
        pass
    else:        
        if verbose == True:
            print("Making new SimpleLoad object")
        load = _SimpleLoad(os.path.join(load_path), series_by=series_by, sequence_label=sequence_label, verbose=verbose)
        np.save(os.path.join(load_path, "VoltageSeries.npy"), load, allow_pickle=True)
        return load
    

