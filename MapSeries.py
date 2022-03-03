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
        for key, measurement in self.data.items():
            measurement.set_clip_range(start_shift, end_shift)
            
        self._find_common(clipped=True)  
        
        
    def quick_stack_view(self, offset_max=100):
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
    

def MapSeries(load_path, verbose=False,
              sequence_label="cellvoltage", fname_delimiter="_",
              series_by="voltage", sort=True):
    """
    
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
    
