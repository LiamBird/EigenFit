import numpy as np

def percentile_vmax(array, percentile=99):
    array = np.array(array).flatten()
    
    finite_array = array[~np.isnan(array)]
    percentile_pos = int(finite_array.shape[0]*(percentile/100))
    vmax = np.sort(finite_array.flatten())[percentile_pos]
    return vmax