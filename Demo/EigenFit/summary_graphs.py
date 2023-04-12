import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from ipywidgets import interact, IntSlider

peak_labels = {202: "Li${_2}$S${_4}$",
               218: "S$_{8}$",
               398: "Li$_2$S$_6$",
               450: "Li$_2$S$_{2}$ or Li$_2$S$_{4}$",
               504: "Li$_2$S$_6$"}

def percentile_vmax(array, percentile=99):
    finite_array = array[np.isfinite(array)]
    array_sort = np.sort(abs(finite_array).flatten())
    if percentile != 100:
        percentile_idx = int(array_sort.shape[0]*percentile/100)
    else:
        percentile_idx = -1
    return array_sort[percentile_idx]

def normalised_colorbar(figure, cmap, return_mappable_only=False,
                        label="Intensity", min_tick=0, max_tick="Max", axis=None, 
                        cbar_x0=0.85, cbar_width=0.05, labelpad=-40):
    """
    If only mappable is needed:
    cbar = plt.colorbar(sm, cax=cbar_ax)
    
    Otherwise:
    create f, ax 
    call normalised_colorbar to add colorbar to f
    * call AFTER plt.tight_layout() to get right shape
    """
    import matplotlib as mpl
    
    if return_mappable_only == True:
        cmap = plt.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        return sm
    
    else:
        if axis == None:
            cbar_y0 = 0.3
            cbar_height = 0.3
        else:
            cbar_y0 = axis.get_position().y0
            cbar_height = axis.get_position().height


        cbar_ax = figure.add_axes((cbar_x0, cbar_y0, cbar_width, cbar_height))
        cbar_ticks = np.array([0, 1], dtype=int)
        cmap = plt.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=cbar_ticks, cax=cbar_ax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([min_tick, max_tick])
        cbar.set_label(label, rotation=90, labelpad=labelpad)

        return cbar


def percentile_vmax_fit_results(fit_results, percentile, peaks_to_show=[218, 202, 398, 450, 504]):
    combined_data = np.array([fit_results.scaled_results[peak][voltage] for peak in peaks_to_show for voltage in fit_results.voltage_labels if peak in fit_results.scaled_results.keys()]).flatten()
    combined_sort = np.sort(combined_data[combined_data>=0])
    vmax = combined_sort[int(percentile/100*combined_sort.shape[0])]
    return vmax

def heatmaps_grid(fit_results, cycle_type, percentile=99, peaks_to_show=[218, 202, 398, 450, 504]):
    left_margin = 0.05
    top_margin = 0.1
    bottom_margin = 0.1
    x_margin = 0.01
    y_margin = 0.01
    cbar_width = 0.05
    
    vmax = percentile_vmax_fit_results(fit_results, percentile)

    width = (1-2*cbar_width-2*x_margin-left_margin-len(peaks_to_show)*x_margin)/len(peaks_to_show)
    height = (1-top_margin-bottom_margin-len(fit_results.voltage_labels)*y_margin)/len(fit_results.voltage_labels)

    import matplotlib as mpl

    plt.rcParams.update({"font.size": 10})
    f, (axes) = plt.subplots(len(fit_results.voltage_labels), len(peaks_to_show),
                             figsize=(len(peaks_to_show), len(fit_results.voltage_labels)));

    if cycle_type == "discharge":
        voltage_series = fit_results.voltage_labels
    else:
        voltage_series = fit_results.voltage_labels[::-1]

    for nvolt, volt in enumerate(voltage_series):
        for npeak, peak in enumerate(peaks_to_show):
            try:
                axes[nvolt, npeak].imshow(fit_results.scaled_results[peak][volt], vmin=0, vmax=vmax, cmap="YlOrRd");
            except:
                pass
            axes[nvolt, npeak].set_xticks([]);
            axes[nvolt, npeak].set_yticks([]);

            axes[nvolt, 0].set_ylabel("{} V".format(volt));

            axes[nvolt, npeak].set_position((left_margin+npeak*(x_margin+width),
                                            bottom_margin+nvolt*(y_margin+height),
                                            width, height));     

    for npeak, peak in enumerate(peaks_to_show):
        f.text(x=left_margin+npeak*(x_margin+width)+0.5*width,
               y=1-top_margin,
               s="{}\n{}{}".format(peak_labels[peak], peak, "cm$^{-1}$"),
               rotation=90, ha="center", );
    cbar_ax = f.add_axes((1-2*(cbar_width+x_margin), 0.3, cbar_width, 0.3));

    cbar_ticks = np.array([0, 1], dtype=int);

    cmap = plt.cm.get_cmap("YlOrRd");
    norm = mpl.colors.Normalize(vmin=0, vmax=1);
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap);
    sm.set_array([]);
    cbar = plt.colorbar(sm, cax=cbar_ax, ticks=cbar_ticks);
    cbar.set_label("Normalised intensity", rotation=90)
    cbar_ax.set_aspect("auto")   
    
    return f, axes;

def get_bplot_positions(peaks_to_show=[218, 202, 398, 450, 504]):
    bplot_widths = 1/(len(peaks_to_show)+1)
    bplot_positions = np.linspace(-0.5+bplot_widths, 0.5-bplot_widths, len(peaks_to_show))
    return bplot_positions
    

def make_boxplot(fit_results, cycle_type, peaks_to_show=[218, 202, 398, 450, 504], figsize=(8, 6)):
    bplot_intensities = {}
    for peak in peaks_to_show:
        try:
            peak_intensity = [fit_results.scaled_results[peak][voltage] for voltage in fit_results.voltage_labels]
            bplot_intensities.update([(peak, 
                                       [voltage_results[voltage_results>0] for voltage_results in peak_intensity])])
        except:
            pass
        
    bplot_widths = 1/(len(peaks_to_show)+1)
    bplot_positions = np.linspace(-0.5+bplot_widths, 0.5-bplot_widths, len(peaks_to_show))
    bplot_colors = dict([(key, plt.cm.tab20(i)) for i, key in enumerate(peaks_to_show)])
    
    f, ax = plt.subplots(figsize=(figsize))
    bplot_boxes = dict([(keys, 
                         ax.boxplot(values,
                            positions=np.arange(len(values))-bplot_positions[peaks_to_show.index(keys)],
                            widths=bplot_widths, patch_artist=True,
                            medianprops=dict(color="black"))) for n, (keys, values) in enumerate(bplot_intensities.items())])

    for keys, values in bplot_boxes.items():
        ax.plot([], [], "s", color=bplot_colors[keys], label=peak_labels[keys])
        for patch in values["boxes"]:
            patch.set_facecolor(bplot_colors[keys])
        for patch in values["fliers"]:
            patch.set_markeredgecolor(bplot_colors[keys])
    ax.legend(ncol=len(peak_labels), loc="lower center", bbox_to_anchor=(0.5, 1))
    ax.set_xticks(np.arange(len(fit_results.voltage_labels)))
    ax.set_xticklabels(fit_results.voltage_labels, rotation=90)

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.grid(which="minor")
    ax.set_yticks([])
    ax.set_ylim([0, None])

    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.grid(which="minor")
    if cycle_type == "discharge":
        ax.invert_xaxis()
    ax.tick_params(which="major", tickdir="in", left=False)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_ylabel("Normalised intensity (a.u.)")
    ax.set_xlabel("Voltage vs. Li/ Li$^{+}$ (V)")

    plt.tight_layout()
    return f, ax


def slider_grid(fit_results, percentile=99):
    import matplotlib as mpl
    from matplotlib.ticker import FormatStrFormatter

    x_coords = fit_results.common.x_coords
    y_coords = fit_results.common.y_coords
    
    ax_grid = np.array([
        [202, 398, 218],
        [450, 504, None],
    #     [398, 504]
    ])

    peaks_to_show = [218, 202, 398, 450, 504]
    peak_labels = {202: "Li${_2}$S${_4}$",
                   218: "S$_{8}$",
                   398: "Li$_2$S$_6$",
                   450: "Li$_2$S$_{2}$ or Li$_2$S$_{4}$",
                   504: "Li$_2$S$_6$"}

    vmax = percentile_vmax_fit_results(fit_results, percentile)
    plt.rcParams.update({"font.size": 12})
    f, (axes) = plt.subplots(2, 3,
                             figsize=(6, 5))
    volt = 2.39

    cbar_width = 0.01
    cbar_ax = axes[1, -1]
    cpos = cbar_ax.get_position()
    cbar_ax.set_position((cpos.x0+0.5*cpos.width-cbar_width,
                          cpos.y0+0.2*cpos.height,
                          cbar_width,
                          cpos.height*0.6))
    cbar_ticks = np.array([0, 1], dtype=int)
    cmap = plt.cm.get_cmap("YlOrRd")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = f.colorbar(sm, ticks=cbar_ticks, cax=cbar_ax)
    cbar.set_ticklabels(["0", "Max"])
    cbar_ax.set_aspect("auto")

    f.text(x=-0.02, y=0.5, s="y pos ($\mu$m)", rotation=90)
    f.text(x=0.5, y=0.01, s="x pos ($\mu$m)", ha="center")
    # for nvolt, volt in enumerate(fit_results.voltage_labels):

    # #     plt.delaxes(axes[0, 0])
    def update(v_idx):
        volt = fit_results.voltage_labels[v_idx]
        for npeak, peak in enumerate(peaks_to_show):
            if peak in fit_results.peak_labels:
                x, y = np.nonzero(ax_grid==peak)
                axes[x[0], y[0]].imshow(fit_results.scaled_results[peak][volt],
                                        extent=(min(x_coords), max(x_coords), min(y_coords), max(y_coords)),
                                        vmin=0, vmax=vmax, cmap="YlOrRd")
                axes[x[0], y[0]].set_title("{}\n{} cm{}".format(peak_labels[peak], peak, "$^{-1}$"))
                if x==0:
                    axes[x[0], y[0]].set_xticks([])
                else:
                    axes[x[0], y[0]].set_xticklabels(axes[x[0], y[0]].get_xticks(),
                                                     rotation=90)
                    axes[x[0], y[0]].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

                if y>0:
                    axes[x[0], y[0]].set_yticks([])
        f.suptitle("{} V".format(volt))
    interact(update, v_idx=IntSlider(min=0, max=len(fit_results.voltage_labels)-1, step=1))
    
    return f, axes



def make_boxplot_real_x(fit_results, cycle_type, skip_volts=[], peaks_to_show=[218, 202, 398, 450, 504],
                        figsize=(18, 6)):
    
    voltages = [v for v in fit_results.voltage_labels if v not in skip_volts]
    positions = [volt+0.5*(voltages[nvolt+1]-voltages[nvolt]) for nvolt, volt in enumerate(voltages[:-1])]
    positions.append(voltages[-1]+0.5*(voltages[-1]-voltages[-2]))
    spacing = np.min(np.array(positions)[1:]-np.array(positions)[:-1])
    width = spacing/(len(peaks_to_show)+1)
    offset = np.linspace(-0.5*spacing, 0.5*spacing, len(peaks_to_show)+2)[1:-1]

    bplot_colors = dict([(key, plt.cm.tab20(i)) for i, key in enumerate(peaks_to_show)])
    bplot_intensities = {}
    for peak in peaks_to_show:
        try:
            peak_intensity = [fit_results.scaled_results[peak][voltage] for voltage in fit_results.voltage_labels if voltage not in skip_volts]
            bplot_intensities.update([(peak, 
                                       [voltage_results[voltage_results>0] for voltage_results in peak_intensity])])
        except:
            pass


    f, ax = plt.subplots(figsize=figsize)
    bplot_boxes = dict([(keys, 
                         ax.boxplot(values,
                      positions=positions+offset[n], 
                       widths=width, patch_artist=True, medianprops=dict(color="white")))
                        for n, (keys, values) in enumerate(bplot_intensities.items())])

    for keys, values in bplot_boxes.items():
        ax.plot([], [], color=bplot_colors[keys], label=peak_labels[keys])
        for patch in values["boxes"]:
            patch.set_facecolor(bplot_colors[keys])
        for patch in values["fliers"]:
            patch.set_markeredgecolor(bplot_colors[keys])

    [ax.axvline(voltage, color="k", lw=0.5) for voltage in fit_results.voltage_labels if voltage not in skip_volts];
    x_ticks_min = 1.5
    x_ticks_max = 2.9
    x_ticks_step = 0.1

    ax.set_xticks(np.arange(x_ticks_min, x_ticks_max, x_ticks_step))
    ax.set_xticklabels(np.arange(x_ticks_min, x_ticks_max, x_ticks_step))
    ax.set_xlim([x_ticks_min, x_ticks_max])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.grid(which="minor")
    ax.set_yticks([])
    ax.set_ylim([0, None])

    ax.set_ylabel("Normalised intensity (a.u.)")
    ax.set_xlabel("Voltage vs. Li/ Li$^{+}$ (V)")
    ax.tick_params(which="major", tickdir="in", left=False)
    ax.tick_params(which="minor", bottom=False, left=False)
    if cycle_type == "discharge":
        ax.invert_xaxis()
        
    ax.legend(ncol=len(peak_labels), loc="lower center", bbox_to_anchor=(0.5, 1))

        
    return f, ax


def make_boxplot_real_y(fit_results, legend_cols=2, skip_volts=[], peaks_to_show=[218, 202, 398, 450, 504],
                        figsize=(6, 12)):
    
    voltages = [v for v in fit_results.voltage_labels if v not in skip_volts]
    positions = [volt+0.5*(voltages[nvolt+1]-voltages[nvolt]) for nvolt, volt in enumerate(voltages[:-1])]
    positions.append(voltages[-1]+0.5*(voltages[-1]-voltages[-2]))
    spacing = np.min(np.array(positions)[1:]-np.array(positions)[:-1])
    width = spacing/(len(peaks_to_show)+1)
    offset = np.linspace(-0.5*spacing, 0.5*spacing, len(peaks_to_show)+2)[1:-1]

    bplot_colors = dict([(key, plt.cm.tab20(i)) for i, key in enumerate(peaks_to_show)])
    bplot_intensities = {}
    for peak in peaks_to_show:
        try:
            peak_intensity = [fit_results.scaled_results[peak][voltage] for voltage in fit_results.voltage_labels if voltage not in skip_volts]
            bplot_intensities.update([(peak, 
                                       [voltage_results[voltage_results>0] for voltage_results in peak_intensity])])
        except:
            pass


    f, ax = plt.subplots(figsize=figsize)
    bplot_boxes = dict([(keys, 
                         ax.boxplot(values,
                      positions=positions+offset[n], vert=False,
                       widths=width, patch_artist=True, medianprops=dict(color="white")))
                        for n, (keys, values) in enumerate(bplot_intensities.items())])

    for keys, values in bplot_boxes.items():
        ax.plot([], [], color=bplot_colors[keys], label=peak_labels[keys])
        for patch in values["boxes"]:
            patch.set_facecolor(bplot_colors[keys])
        for patch in values["fliers"]:
            patch.set_markeredgecolor(bplot_colors[keys])

    [ax.axhline(voltage, color="k", lw=0.5) for voltage in fit_results.voltage_labels if voltage not in skip_volts];
    y_ticks_min = 1.5
    y_ticks_max = 2.9
    y_ticks_step = 0.1

    ax.set_yticks(np.arange(y_ticks_min, y_ticks_max, y_ticks_step))
    ax.set_yticklabels(np.arange(y_ticks_min, y_ticks_max, y_ticks_step))
    ax.set_ylim([y_ticks_min, y_ticks_max])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.xaxis.grid(which="minor", alpha=.5)
    ax.xaxis.grid(which="major", alpha=1, lw=1)
    ax.set_xticks([])
    ax.set_xlim([0, None])

    ax.set_xlabel("Normalised intensity (a.u.)")
    ax.set_ylabel("Voltage vs. Li/ Li$^{+}$ (V)")
    ax.tick_params(which="major", tickdir="in")# , left=False)
    ax.tick_params(which="minor", tickdir="in", bottom=True, top=True) #left=False)
        
    ax.legend(ncol=legend_cols, loc="lower center", bbox_to_anchor=(0.5, 1))

        
    return f, ax