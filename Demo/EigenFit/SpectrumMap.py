"""
Spectrum Map class
Version 2
Last updated: 19/10/2021
Created: 08/10/2021
Added to EigenFit: 06/01/2022
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

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
    
    return np.vstack((shift, repaired)).T


class _Map(object):
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
    def __init__(self, shift, map_data):
        self.shift = shift
        self.shift_extent = shift.shape[0]
        self.map = map_data
        
        
class SpectrumMap(object):
    """
    A class to contain Raman map data, with associated pre-processing and display functions
    ...
    Input arguments
    ----------
    filename: str
        name and path of the four-column text file of Raman map data, including file extension
    Attributes
    ----------
    x_coords: numpy array
        x positions corresponding to spectra positions
    y_coords: numpy array
        y positions corresponding to spectra positions
    x_extent: int
        number of x positions at which spectra were collected
    y_extent: int
        number of y positions at which spectra were collected
        
    Subclass (_Map) for raw data: 
    raw.shift: numpy array
        the Raman shift axis corresponding to the map data
    raw.shift_extent: int
        number of shift values
    raw.map: numpy array
        3d array containing the map data (dimensions for x, y, and shift axes)
        
    Methods
    ----------
    slide_viewer(xmin=None, ymin=None):
        returns an interactive (ipywidgets) plot of the signal
        
    fix_cosmic_ray(x_pos, y_pos, spike_pos, smooth_width=10):
        repairs cosmic rays at user-defined x_pos, y_pos map positions at shift value spike_pos. 
        corrects values directly in all _Map subclasses (raw, plus ccd and clipped if created)
        returns none
    
    set_CCD_remove(pixel_position, smooth_width=10):
        removes a spike located at the same shift value (i.e. pixel position) throughout a map
        returns 'ccd' as new attribute of SpectrumMap (_Map class with shift, shift_extent, and map attributes)
        
    set_clip_range(start_shift=None, end_shift=None):
        returns 'clipped' as a new attribute of SpectrumMap (_Map class with shift, shift_extent, and map attributes)
        
    Notes
    ----------
    
     To instantiate the class, use:
            s = SpectrumMap(filename)
    The data will be accessible using:
    s.raw.map
    To view the data, use:
        s.slide_viewer()
    To remove cosmic rays, use:
        s.fix_cosmic_ray(x_pos, y_pos, spike_pos)
    To remove a consistent spike (from a dead CCD pixel), use:
        s.set_CCD_remove(pixel_position)
        The fixed data will be accessible using:
        s.ccd.map
        NB: fix the spike before clipping the data range
    To clip the range of data to view, use:
        s.set_clip_range(start_shift, end_shift)
        The clipped data will be accessible using:
        s.clipped.map
    """
    
    def __init__(self, data_file):
        raw_data = np.loadtxt(data_file)
        self.x_coords = np.array(np.unique(raw_data[:, 0]), dtype=int)
        self.y_coords = np.array(np.unique(raw_data[:, 1]), dtype=int)
        raw_shift, raw_shift_locs = np.unique(raw_data[:, 2], return_index=True)
        
        self.x_extent = self.x_coords.shape[0]
        self.y_extent = self.y_coords.shape[0]        
        
        raw_shift_extent = raw_shift.shape[0]
        raw_map = np.zeros((self.x_extent, self.y_extent, raw_shift_extent))
        for nx, x in enumerate(self.x_coords):
            for ny, y in enumerate(self.y_coords):
                spectrum = raw_data[np.nonzero((raw_data[:, 0] == x) & (raw_data[:, 1] == y)), -1].flatten()
                raw_map[nx, ny, :] = spectrum[::-1] 
                
        setattr(self, "raw", _Map(shift=raw_shift, map_data=raw_map))
        
    def set_CCD_remove(self, pixel_position, smooth_width=10):
        """
        Removes a spike at a constant shift value at every (x, y) position in a Raman map, for example due to a faulty pixel on a CCD device
        
            Parameters:
                pixel_position (float): The shift value at which the spike occurs
                smooth_width (int, default=10): The number of shift values to be replaced by random noise each side of the spike
                
            Returns:
                self.ccd (object): A _Map class with shift, shift_extent, and map attributes
        """
        pixel_idx = np.argmin(abs(self.raw.shift-pixel_position))               
        CCD_map = np.zeros((self.x_extent, self.y_extent, self.raw.shift_extent))
        
        for x in range(self.x_extent):
            for y in range(self.y_extent):
                pre_max = np.nanmax(self.raw.map[x, y, pixel_idx-2*smooth_width:pixel_idx-smooth_width])
                post_max = np.nanmax(self.raw.map[x, y, pixel_idx+smooth_width:pixel_idx+2*smooth_width])
                pre_min = np.nanmin(self.raw.map[x, y, pixel_idx-2*smooth_width:pixel_idx-smooth_width])
                post_min = np.nanmin(self.raw.map[x, y, pixel_idx+smooth_width:pixel_idx+2*smooth_width])
                
                noise_fill = min([pre_min, post_min])+(max([pre_max, post_max])-min([pre_min, post_min]))*np.random.rand(2*smooth_width)
                
                CCD_map[x, y, :pixel_idx-smooth_width] = self.raw.map[x, y, :pixel_idx-smooth_width]              
                CCD_map[x, y, pixel_idx-smooth_width:pixel_idx+smooth_width] = noise_fill
                CCD_map[x, y, pixel_idx+smooth_width:] = self.raw.map[x, y, pixel_idx+smooth_width:] 
                
        setattr(self, "ccd", _Map(shift=self.raw.shift, map_data=CCD_map))
        
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
                
        if "ccd" in vars(self).keys():
            setattr(self, "clipped", _Map(shift=self.raw.shift[start_idx:end_idx],
                                          map_data=self.ccd.map[:, :, start_idx:end_idx]))
        else:
            setattr(self, "clipped", _Map(shift=self.raw.shift[start_idx:end_idx],
                                          map_data=self.raw.map[:, :, start_idx:end_idx]))
    
    def fix_cosmic_ray(self, x_pos, y_pos, spike_pos, smooth_width=10):
        """
        Repairs cosmic rays identified by the user and applies repairs to "raw", AND "ccd", AND "clipped" maps if defined. 
        
            Parameters:
                x_pos (int): the x position of the spectrum to repair within the SpectrumMap map
                y_pos (int): the y position of the spectrum to repair within the SpectrumMap map
                spike_pos (float): the shift value at which the spike occurs
                smooth_width (int, default=10): the number of shift positions either side of the spike to replace with noise
        
        """
        maps_to_fix = [key for key in vars(self).keys() if key in ["raw", "ccd", "clipped"]]
        
        for processed in maps_to_fix:
            if spike_pos < max(vars(self)[processed].shift) and spike_pos > min(vars(self)[processed].shift):
                vars(self)[processed].map[x_pos, y_pos, :] = CRE_single(shift=vars(self)[processed].shift,
                                                                       intensity=vars(self)[processed].map[x_pos, y_pos, :],
                                                                       spike_pos=spike_pos,
                                                                       smooth_width=smooth_width)[:, 1]
    def slide_viewer(self, display="raw"):
        """
        Returns an ipywidgets controlled interactive graph display, with widgets to control x and y position of spectrum to display
            
            Parameters:
                display (str): one of "raw", "clipped", or "ccd"
                               selects the (pre-processed) data to display
        """
        f, ax = plt.subplots()
        shift = vars(self)[display].shift
        map_data = vars(self)[display].map
        
        spectrum, = ax.plot(shift, map_data[0, 0, :])
        
        ax.tick_params(which="both", tickdir="in", right=True, top=True)
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        
        def update(x, y):
            spectrum.set_ydata(map_data[x, y, :])
            ax.set_ylim([0.9*min(map_data[x, y, :]),
                         1.1*max(map_data[x, y, :])])

        
        interact(update, x=IntSlider(min=0, max=self.x_extent-1, step=1),
                         y=IntSlider(min=0, max=self.y_extent-1, step=1))