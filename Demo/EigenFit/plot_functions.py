import numpy as np
import EigenFit as eig
from matplotlib.ticker import MultipleLocator

def discharge_voltages(cycle, voltage_list):
    voltages = np.sort(voltage_list)[::-1]
    return (2*cycle-1)*(volt_max-volt_min)-(voltages-volt_min)

def charge_voltages(cycle, voltage_list):
    voltages = np.sort(voltage_list)
    return (2*cycle-1)*(volt_max-volt_min)+(voltages-volt_min)

def make_intensities(sample, cycle):
    all_peaks = np.unique(np.hstack([[*voltage_values.keys()] for voltage_values in vars(sample)[cycle].values.values()]))
    x_extent = sample.raw_data[cycle].common.x_extent
    y_extent = sample.raw_data[cycle].common.y_extent
    
    intensities = {}
    for voltage, voltage_values in vars(sample)[cycle].values.items():
        intensities.update([(voltage, {})])
        for peak in all_peaks:
            if peak in voltage_values.keys():
                intensities[voltage].update([(peak,
                                              voltage_values[peak]["amplitude"]/voltage_values[peak]["sigma"]/np.pi)])
            else:
                intensities[voltage].update([(peak, np.full((x_extent, y_extent), np.nan))])
                
    return intensities

def make_bplot_intensities(intensity_data, predict_dict, discharge=True, peak_plots=[218, 398, 450]):
    
    if discharge==True:
        voltages = np.sort([*intensity_data.keys()])[::-1]
    else:
        voltages = np.sort([*intensity_data.keys()])
        
    bp_i = dict([(peak, dict([(voltage, np.full((x_extent, y_extent), np.nan)) 
                              for voltage in voltages]))
                for peak in peak_plots])
        
    for peak in peak_plots:
        for voltage in voltages:
            if peak in np.sort([*intensity_data[voltage].keys()])[::-1]:
                bp_i[peak][voltage] = intensity_data[voltage][peak][predict_dict[voltage]==1][
                                np.isfinite(intensity_data[voltage][peak][predict_dict[voltage]==1])
                ]
    return bp_i


def PS_peak_scatter(sample, cycle, predict_dict, discharge=True, peak_plots=[398, 450], voltage_cutoff=2.1):

    all_peaks = np.unique(np.hstack([[*voltage_values.keys()] for voltage_values in vars(sample)[cycle].values.values()]))
    x_extent = sample.raw_data[cycle].common.x_extent
    y_extent = sample.raw_data[cycle].common.y_extent
    
    intensities = make_intensites(cycle)
                
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
    
    peak_dict = {218: "S$_{8}$",
             202: "S$_{4}^{2-}$",
             398: "I(398)",##"S$_{6, 8}^{2-}$",
             450: "I(450)", ##"S$_{4, 2}^{2-}$",
             504: "S$_{6}^{2-}$"}
    
    vmax = eig.percentile_vmax(np.hstack((values[peak].flatten() for values in intensities.values()
                                          for peak in values.keys() if peak in peak_plots)))
    
    f, ax = plt.subplots()

    ax.set_prop_cycle("marker", markers)

    for nvolt, voltage in enumerate([*intensity_filtered.keys()][::-1]):
        ax.plot(intensities[voltage][peak_plots[0]][predict_dict[voltage]==1],
                    intensities[voltage][peak_plots[1]][predict_dict[voltage]==1],
                    label=voltage, color=colors[nvolt], ls="none")

    ax.set_xlabel("{} (a.u.)".format(peak_dict[peak_plots[0]]))
    ax.set_ylabel("{} (a.u.)".format(peak_dict[peak_plots[1]]))

    x_min = ax.get_xlim()
    y_min = ax.get_ylim()
    ax.plot([x_min[0], x_min[1]], [y_min[0], y_min[1]],
            color="k", zorder=0, marker=None, lw=0.5)
    ax.set_xlim(x_min)
    ax.set_ylim(y_min)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_aspect("equal")
    
    return f, ax



def make_intensities(cycle):
    import numpy as np
    all_peaks = np.unique(np.hstack([[*voltage_values.keys()] for voltage_values in vars(sample)[cycle].values.values()]))
    x_extent = sample.raw_data[cycle].common.x_extent
    y_extent = sample.raw_data[cycle].common.y_extent
    
    intensities = {}
    for voltage, voltage_values in vars(sample)[cycle].values.items():
        intensities.update([(voltage, {})])
        for peak in all_peaks:
            if peak in voltage_values.keys():
                intensities[voltage].update([(peak,
                                              voltage_values[peak]["amplitude"]/voltage_values[peak]["sigma"]/np.pi)])
            else:
                intensities[voltage].update([(peak, np.full((x_extent, y_extent), np.nan))])
                
    return intensities