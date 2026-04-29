import numpy as np

class _Spectrum(object):
    """
    Hidden container class
    ...
    Attributes
    ----------
    shift: array
        The shift (x) axis
   shift_extent: int
        The length of the shift axis array
   map: array
        The Raman map data with size (x_extent, y_extent, shift_extent)
    """
    def __init__(self, shift, spectrum_data):
        self.shift = shift
        self.shift_extent = shift.shape[0]
        self.spectrum = spectrum_data


class _noise_replace(object):
    """
    Hidden class with attributes for replacing spikes with noise
    ...
    
    Attributes
    ----------
    noise_fill : array
        a short array containing noise within the range of minimum/ maximum values of surrounding signal
        
    start_replace : float
        the index to start replacing data with noise
        
    end_replace : float
        the index to finish replacing data with noise    
    """
    def __init__(self, shift, intensity, spike_pos, smooth_width=10):
        spike_idx = np.argmin(abs(shift-spike_pos))
        ## if near the minimum of the shift axis:
        if spike_idx < 2*smooth_width:
            min_spike = 0
            min_avg = 0
            max_avg = spike_idx+2*smooth_width
            max_spike = spike_idx+smooth_width
        ## if near the maximum of the shift axis:
        elif shift.shape[0]-spike_idx < 2*smooth_width:
            min_spike = spike_idx-smooth_width
            min_avg = spike_idx-2*smooth_width
            max_avg = shift.shape[0]
            max_spike = shift.shape[0]
        else:
            min_spike = spike_idx-smooth_width
            min_avg = spike_idx-2*smooth_width
            max_avg = spike_idx+2*smooth_width
            max_spike = spike_idx+smooth_width

        noise_min = np.nanmin([np.nanmin(intensity[min_avg:min_spike]),
                      np.nanmin(intensity[max_spike:max_avg])])

        noise_max = np.nanmax([np.nanmax(intensity[min_avg:min_spike]),
                      np.nanmax(intensity[max_spike:max_avg])])

        self.noise_fill = np.random.rand(2*smooth_width)*(noise_max-noise_min)+noise_min
        self.start_replace = min_spike
        self.end_replace = max_spike

def CRE_single(shift, intensity, spike_pos, smooth_width=10):
    """
    Function to repair a single spectrum with a cosmic ray spike at a user-defined position
    
        Parameters:
            shift (array): The shift/ x-axis for the data to be repaired
            intensity (array): The intensity/ y values of the data to be repaired
            spike_pos (float): The shift value at which the cosmic ray occurs
            smooth_width (int) (default = 10): The number of shift values to replace by noise
    """
    repaired = np.zeros((intensity.shape))
    
    if type(spike_pos)==list:
        replacements = [_noise_replace(shift=shift, intensity=intensity, spike_pos=pos, smooth_width=smooth_width) \
                        for pos in spike_pos]
        replacement_idx = np.array([np.arange(replacement.start_replace, replacement.end_replace) for replacement in replacements])
        
        for x in range(intensity.shape[0]):
            if x not in replacement_idx:
                repaired[x] = intensity[x]
                
        for rp in replacements:
            repaired[rp.start_replace:rp.end_replace] = rp.noise_fill
            repaired[:rp.start_replace] = intensity[:rp.start_replace]
            repaired[rp.end_replace:] = intensity[rp.end_replace:]

    if type(spike_pos) == float or type(spike_pos) == int:
        rp = _noise_replace(shift=shift, intensity=intensity, spike_pos=spike_pos, smooth_width=smooth_width)
        repaired[rp.start_replace:rp.end_replace] = rp.noise_fill
        repaired[:rp.start_replace] = intensity[:rp.start_replace]
        repaired[rp.end_replace:] = intensity[rp.end_replace:]
    
#     return np.vstack((shift, repaired)).T
        return repaired


class Spectrum(object):
    def __init__(self, filename, delimiter=","):
        dataload = np.loadtxt(filename, delimiter=delimiter)
        self.raw = _Spectrum(dataload[:, 0].flatten(),
                             dataload[:, 1].flatten())

        self._version = "03.10.2024"
        self._change_log = ["03.10.2024: Added delimiter argument. (Started collecting data from Horiba instrument with \t delimiter)"]
        
    def set_clip_range(self, start_shift=None, end_shift=None):
        """
        Returns a map with x_extent and y_extent corresponding to raw data, with a range of shift values corresponding to a region of interest defined by the user
        
            Parameters:
                start_shift (float, default=None): The shift value corresponding to the lower bound of the region of interest.
                                                   If 'None', uses the first shift value in the data
                end_shift (float, default=None): The shift value corresponding to the upper bound of the region of interest.
                                                 If 'None', uses the last shift value in the data
        """
        if start_shift == None:
            start_shift = min(self.raw.shift)
        if end_shift == None:
            end_shift = max(self.raw.shift)
        start_idx = np.argmin(abs(self.raw.shift-start_shift))
        end_idx = np.argmin(abs(self.raw.shift-end_shift))
        setattr(self, "clipped", _Spectrum(shift=self.raw.shift[start_idx:end_idx],
                                          spectrum_data=self.raw.spectrum[start_idx:end_idx]))
        
        
    def set_baseline_removal(self, degree=4):
        from BaselineRemoval import BaselineRemoval
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        
        if "clipped" in vars(self):
            data_type = "clipped"
        else:
            data_type = "raw"
        bkg_rem = BaselineRemoval(vars(self)[data_type].spectrum).IModPoly(degree)
        setattr(self, "bkg_rem", _Spectrum(shift=vars(self)[data_type].shift,
                                           spectrum_data=bkg_rem))
        
    def set_spike_remove(self, spike_pos, smooth_width=10):
        for data_type in ["raw", "clipped", "bkg_rem"]:
            if data_type in vars(self):
                vars(self)[data_type].spectrum = CRE_single(vars(self)[data_type].shift,
                                                            vars(self)[data_type].spectrum,
                                                            spike_pos, smooth_width=smooth_width)
