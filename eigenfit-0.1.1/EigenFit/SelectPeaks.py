import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import ipywidgets as wg

class SelectPeaks(object):
    def __init__(self, voltage_series, peaks_to_fit=[152, 202, 218, 248, 398, 438, 450, 473, 504, 745], reload=False):
        layout = layout=wg.Layout(flex="1 1 0%", width="auto")
        
        if reload!=False:
            self._previous_values = np.load(reload, allow_pickle=True).item()
        else:
            self._previous_values = dict([(volt,
                                     dict([(peak, False) for peak in peaks_to_fit])) for volt in voltage_series
                                   ])
        self._checkbox_dict = {}
        self._voltages = voltage_series
        self._peaks = peaks_to_fit
        
        for volt in voltage_series:
            self._checkbox_dict.update([(volt, {})])
            if volt in self._previous_values.keys():
                for peak in peaks_to_fit:
                    if peak in self._previous_values[volt].keys():
                        self._checkbox_dict[volt].update([(peak, wg.Checkbox(indent=False, value=self._previous_values[volt][peak], layout=layout))])
                    else:
                        self._checkbox_dict[volt].update([(peak, wg.Checkbox(indent=False, value=False, layout=layout))])
            else:
                for peak in peaks_to_fit:
                    self._checkbox_dict[volt].update([(peak, wg.Checkbox(indent=False, value=False, layout=layout))])
        
        self.grid = wg.HBox([wg.VBox([wg.Label(value=str(peak)) for peak in [" "]+[*self._checkbox_dict[volt].keys()]])]+\
                            [wg.VBox([wg.Label(value=str(volt))]+[*self._checkbox_dict[volt].values()]) for volt in voltage_series])
    def get_checkbox_values(self, save_name=None, save_path=None):
        if save_path == None or os.path.isdir(save_path)==False:
            try:
                os.mkdir("selected_peaks")
            except:
                pass
            save_path = "selected_peaks"
        
        self.saved_values = dict([(float(self.grid.children[nvolt].children[0].value),
                         dict([(int(peak), self.grid.children[nvolt].children[npeak+1].value) for npeak, peak in enumerate([label.value for label in self.grid.children[0].children[1:]])])
                        ) for nvolt in range(1, len(self.grid.children))])
        
        if save_name != None:
            np.save(os.path.join(save_path, "{}.npy".format(save_name)),
                    self.saved_values,
                    allow_pickle=True)
            
    def _get_positions(self):
        self._values_grid = np.array([[*values.values()] for values in self.saved_values.values()])
        