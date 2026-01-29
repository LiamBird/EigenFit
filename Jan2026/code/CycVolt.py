import numpy as np
import re
import matplotlib.pyplot as plt


class CycVolt(object):
    """
    Loads cyclic voltammetry data sorted by cathodic/ anodic cycle and provides plotting tool
    
    Attributes:
    ----------
    n_cycles: int
        number of cycles (cathodic+anodic sweeps)
        
    scan_rate: float
        the CV sweep rate (mV/s)
    
    anodic: subclass with attributes 'voltage', 'current', and 'time' 
        each of 'voltage', 'current', and 'time' is a list with length n_cycles, where each element is the data from one cycle
    
    cathodic: subclass with attributes 'voltage', 'current', and 'time'
        same as 'anodic'
    
    """
    
    def __init__(self, filename, tolerance=0.1):
        with open(filename) as f:
            for line in f.readlines():
                if "dE/dt" in line and "unit" not in line:
                    self.scan_rate = float(re.findall("\d+.\d+", line)[0])
                    
        data = mpt_to_df(filename)
        voltage = np.array(data["Ewe/V"])
        dV = voltage[1:]-voltage[:-1]
        current = np.array(data["<I>/mA"])
        charge_idx = np.argwhere(np.sign(dV)==1).flatten()
        discharge_idx = np.argwhere(np.sign(dV)==-1).flatten()
        time = np.array(data["time/s"])
        
        discharge_ends_WITH_super = [num for n, num in enumerate(discharge_idx[:-1]) if discharge_idx[n+1]-discharge_idx[n]!=1]
        discharge_ends = [num for num in discharge_ends_WITH_super if any([abs(voltage[num+1]-limit)<tolerance for limit in [np.min(voltage), np.max(voltage)]])]
        
        charge_ends_WITH_super = [num for n, num in enumerate(charge_idx[:-1]) if charge_idx[n+1]-charge_idx[n]!=1]
        charge_ends = [0]+[num for num in charge_ends_WITH_super if any([abs(voltage[num+1]-limit)<tolerance for limit in [np.min(voltage), np.max(voltage)]])]
        
        if discharge_ends[-1]>charge_ends[-1]:
            n_cycles = len(discharge_ends)
            charge_ends += [charge_idx[-1]]
        else:
            n_cycles = len(charge_ends)
            discharge_ends += [discharge_idx[-1]]
            
        self.n_cycles = n_cycles
            
        class _State(object):
            """ Hidden subclass for easy access to voltage, current, and time data for anodic and cathodic sweeps """
            def __init__(state_self, state):
                
                ## Patch
                state_self.voltage = []
                state_self.current = []
                state_self.time = []
                ## Patch
                
                if state == "discharge":
                    ## Patch
                    for n in range(len(discharge_ends)):
                        try:
                            state_self.voltage.append(voltage[charge_ends[n]:discharge_ends[n]])
                            state_self.current.append(current[charge_ends[n]:discharge_ends[n]])
                            state_self.time.append(time[charge_ends[n]:discharge_ends[n]])
                        except:
                            print("Failed discharge cycle {}".format(n))
                    ## Patch

#                     state_self.voltage = [voltage[charge_ends[n]:discharge_ends[n]] for n in range(len(discharge_ends))]
#                     state_self.current = [current[charge_ends[n]:discharge_ends[n]] for n in range(len(discharge_ends))]
#                     state_self.time = [time[charge_ends[n]:discharge_ends[n]] for n in range(len(discharge_ends))]
                elif state == "charge":
                    ## Patch
                    for n in range(len(charge_ends)-1):
                        try:
                            state_self.voltage.append(voltage[discharge_ends[n]:charge_ends[n+1]])
                            state_self.current.append(current[discharge_ends[n]:charge_ends[n+1]])
                            state_self.time.append(time[discharge_ends[n]:charge_ends[n+1]])
                        except:
                            print("Failed charge cycle {}".format(n))
                    ## Patch
                
#                     state_self.voltage = [voltage[discharge_ends[n]:charge_ends[n+1]] for n in range(len(charge_ends)-1)]
#                     state_self.current = [current[discharge_ends[n]:charge_ends[n+1]] for n in range(len(charge_ends)-1)]
#                     state_self.time = [time[discharge_ends[n]:charge_ends[n+1]] for n in range(len(charge_ends)-1)]
        setattr(self, "cathodic", _State("discharge"))
        setattr(self, "anodic", _State("charge"))
        
    def plot_cv(self, figsize=(8, 8), cycles_to_plot="all", cycles_to_skip="none", cmap="tab20", category_cmap=True, reverse_scan_order=True):
        """
        Plot the cyclic voltammogram (colour-coded by cycle number)
        
        Parameters:
        ----------
        figsize: optional, default = (8, 8)
            size of the figure in inches
            
        cycles_to_plot: {"all", "none", list}, optional, default="all"
            if "all", plots all cycles in range 0 to self.n_cycles
            if "none", plots all cycles if not in list cycles_to_skip
            otherwise provide list of integer indices of cycles to plot (starting at 0)
        
        cycles_to_skip: {"none", list}, optional, default="none"
            if "none", plots all cycles in range 0 to self.n_cycles
            otherwise provide list of integer indices of cycles to plot (starting at 0)
        
        cmap: string, matplotlib.pyplot color map, default = "tab20"
            color map to use for separate cycles
            
        category_cmap: bool, default=True
            using a category colormap for separate cycles
            for >20 cycles on a single plot, the color map may run out of unique colors. Alternatively you may wish to have a color gradient to directly indicate the time sequence of cycles (e.g. light to dark). If so, set category_cmap=False and use a continuous cmap (e.g. cmap="viridis")
            
        reverse_scan_order: bool, default=True
            plots later scans in front of earlier scans (useful where current decreases with cycle number)
      
        
        Returns:
        ----------
        fig, ax: figure and axis with CV plot
        
        """
        f, ax = plt.subplots(figsize=figsize)
        
        if cycles_to_plot == "all":
            n_cycles = list(np.arange(0, self.n_cycles))
        else:
            n_cycles = cycles_to_plot
            
        if cycles_to_skip == "none":
            n_cycles = n_cycles
        else:
            n_cycles = [cyc for cyc in n_cycles if cyc not in cycles_to_skip]
        
        if category_cmap==True:
            plot_colors = [vars(plt.cm)[cmap](i) for i in range(len(n_cycles))]
            for ncycle, cycle in enumerate(n_cycles):
                ax.plot([], [], color=plot_colors[ncycle], label=cycle+1)
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            plot_colors = [vars(plt.cm)[cmap](i) for i in range(np.linspace(0, 1, len(n_cycles)))]
            ax.set_position((0.15, 0.15, 0.75, 0.75))
            cbar_ax = f.add_axes((0.8, 0.15, 0.05, 0.6))
            cbar_ticks = np.array([0, 1], dtype=int)
            cmap = plt.cm.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ticks=cbar_ticks, cax=cbar_ax)
            cbar.set_label("Cycle number", rotation=90)
            cbar.set_ticks([1, max(n_cycles)])
            cbar.set_ticklabels([1, max(n_cycles)])
            
            
        ax.tick_params(which="both", tickdir="in", right=True, top=True)
        ax.set_xlabel("Potential vs. Li/ Li$^{+}$ (V)")
        ax.set_ylabel("Current (mA)")
        ax.axhline(0, color="k", lw=0.5)
        
        for ncycle, cycle in enumerate(n_cycles):
            if reverse_scan_order == True:
                z_order = self.n_cycles-ncycle
            else:
                z_order = ncycle
            try:
                ax.plot(self.discharge.voltage[cycle], self.discharge.current[cycle], color=plot_colors[ncycle], zorder=z_order)
                ax.plot(self.charge.voltage[cycle], self.charge.current[cycle], color=plot_colors[ncycle], zorder=z_order)
            except:
                continue
                    
#         ax.set_xlim([1.5, 2.8])
        
        plt.tight_layout()
        
        return f, ax