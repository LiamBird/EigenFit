import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

def CRE_stacks(map_series, x_seg=7):
    n_rows = int(np.ceil(map_series.common.x_extent/x_seg))

    grid_pos = np.full((x_seg*n_rows), np.nan, dtype=int)
    for y in range(map_series.common.y_extent):
        grid_pos[y] = int(y)

    grid_pos = grid_pos.reshape((x_seg, n_rows)).T
    
    volt = vars(map_series)[map_series._series_by][0]
    
    markers = []
    for idx in range(len(vars(map_series)[map_series._series_by])):
        if idx < 10:
            markers.append("-o")
        elif idx >= 10 and idx < 20:
            markers.append("-s")
        elif idx >=20 and idx < 30:
            markers.append("-^")
    
    f, (axes) = plt.subplots(n_rows, 1, figsize=(6, 2*n_rows))
    
    for ax in axes:
        ax.set_prop_cycle("color", [plt.cm.tab20(i) for i in range(len(vars(map_series)[map_series._series_by]))])
        pos = ax.get_position()
        ax.set_position((pos.x0, pos.y0, 0.8*pos.width, 0.9*pos.height))
    def update(x, seg_id):
        for idx in range(n_rows):
            if np.sign(grid_pos[idx, seg_id]) >= 0:
                try:
                    axes[idx].cla()
                    axes[idx].set_title("x: {}, y: {}".format(x, grid_pos[idx, seg_id]))
                    [axes[idx].plot(map_series.common.shift, map_series.data[volt].clipped.map[x, grid_pos[idx, seg_id]], markers[nvolt], label=volt) for nvolt, volt in enumerate(vars(map_series)[map_series._series_by])];
                except:
                    pass
                
        handles, labels = axes[0].get_legend_handles_labels()
        f.legend(handles, labels, loc="center left", bbox_to_anchor=(0.8, 0.5))
    interact(update, x=IntSlider(min=0, max=map_series.common.x_extent-1, step=1),
                     seg_id=IntSlider(min=0, max=x_seg-1, step=1))
    
    return f, axes