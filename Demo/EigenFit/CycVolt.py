import sys
try:
    [sys.path.remove(name for name in sys.path if "SpectroscoPy" in name)]
except:
    pass

import numpy as np
from mpt_to_df import mpt_to_df

class cvPeaks(object):
    def __init__(peaks_self, cv, t_step=10, intercept_range=5):
        for key in ["_EH0_capacity_pos", "_EH0_capacity_neg", "EH0_capacity", "_EL0_capacity_pos", "_EL0_capacity_neg", "EL0_capacity"]:
            setattr(peaks_self, key, [])
            
        for hour in range(cv.n_cycles):
            try:
                time_d = cv.discharge.time[hour]
                current_d = cv.discharge.current[hour]
                voltage_d = cv.discharge.voltage[hour]

                dtdt = time_d[1:]-time_d[:-1]
                eis_idx = np.argwhere(dtdt>2*np.nanmedian(dtdt)).flatten()

                if eis_idx.shape[0] > 0:
                    current_d[eis_idx[0]:eis_idx[0]+1] = 0

                dr_idx = np.argwhere(np.sign(current_d)==-1)
                dr_time = time_d[dr_idx]
                dr_current = current_d[dr_idx]
                dr_voltage = voltage_d[dr_idx]

                didv = (current_d[t_step:]-current_d[:-t_step])/(voltage_d[t_step:]-voltage_d[:-t_step])
                v_dv = voltage_d[int(t_step):int(t_step)+didv.shape[0]]

                current_maxima = [[x] for x in range(intercept_range, didv.shape[0]-intercept_range) if 
                     all(np.sign(didv[x-intercept_range:x])==-1) and all(np.sign(didv[x:x+intercept_range])==1)]

                current_maxima_in_range = []
                for m in current_maxima:
                    if m > np.nanmin(dr_idx) and m < np.nanmax(dr_idx):
                        if eis_idx.shape[0] > 0 and m != eis_idx and m != eis_idx+1:
                            current_maxima_in_range.append(m[0])
                        else:
                            current_maxima_in_range.append(m[0])

                EH0_start = np.nanmin(dr_idx.flatten())
                EH0_end = current_maxima_in_range[np.argmin(abs(current_maxima_in_range-EH0_start))]

                EL0_end = np.nanmax(dr_idx.flatten())
                EL0_start = current_maxima_in_range[np.argmin(abs(current_maxima_in_range-EL0_end))]
                EH0_capacity_pos = np.trapz(y=current_d[EH0_start:EH0_end], x=time_d[EH0_start:EH0_end]/60/60)
                EL0_capacity_pos = np.trapz(y=current_d[EL0_start:EL0_end], x=time_d[EL0_start:EL0_end]/60/60)

                ## charge
                current_c = cv.charge.current[hour]
                voltage_c = cv.charge.voltage[hour]
                time_c = cv.charge.time[hour]

                c_EL0_idx = np.arange(np.argmin(abs(voltage_c-voltage_d[EL0_end])), np.argmin(abs(voltage_c-voltage_d[EL0_start])))
                voltage_c_EL0 = voltage_c[c_EL0_idx]
                current_c_EL0 = current_c[c_EL0_idx]
                time_c_EL0 = time_c[c_EL0_idx]

                c_EH0_idx = np.arange(np.argmin(abs(voltage_c-voltage_d[EH0_end])), np.argmin(abs(voltage_c-voltage_d[EH0_start])))
                voltage_c_EH0 = voltage_c[c_EH0_idx]
                current_c_EH0 = current_c[c_EH0_idx]
                time_c_EH0 = time_c[c_EH0_idx]

                c_EL0_neg = np.argwhere(np.sign(current_c_EL0)==-1).flatten()
                c_EH0_neg = np.argwhere(np.sign(current_c_EH0)==-1).flatten()

                EH0_capacity_neg = np.trapz(y=current_c_EH0[c_EH0_neg], x=time_c_EH0[c_EH0_neg]/60/60)
                EL0_capacity_neg = np.trapz(y=current_c_EL0[c_EL0_neg], x=time_c_EL0[c_EL0_neg]/60/60)
        
                peaks_self._EH0_capacity_pos.append(EH0_capacity_pos)
                peaks_self._EH0_capacity_neg.append(EH0_capacity_neg)
                peaks_self.EH0_capacity.append(EH0_capacity_pos-EH0_capacity_neg)

                peaks_self._EL0_capacity_pos.append(EL0_capacity_pos)
                peaks_self._EL0_capacity_neg.append(EL0_capacity_neg)
                peaks_self.EL0_capacity.append(EL0_capacity_pos-EL0_capacity_neg)
            except:
                for key in ["_EH0_capacity_pos", "_EH0_capacity_neg", "EH0_capacity", "_EL0_capacity_pos", "_EL0_capacity_neg", "EL0_capacity"]:
                    vars(peaks_self)[key]


class CycVolt(object):
    def __init__(self, filename, tolerance=0.1):
#         print("Using correct function")
        import re
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
        setattr(self, "discharge", _State("discharge"))
        setattr(self, "charge", _State("charge"))
        
    def plot_cv(self, figsize=(8, 8), cycles_to_plot="all", cycles_to_skip="none", cmap="tab20", category_cmap=True, reverse_scan_order=True):
        import matplotlib.pyplot as plt
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
    
    def peak_capacity_ratio(self, t_step=10, intercept_range=5):
        setattr(self, "peak_ratio", cvPeaks(cv=self, t_step=t_step, intercept_range=intercept_range))