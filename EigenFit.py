import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({"xtick.direction": "in"})
plt.rcParams.update({"ytick.direction": "in"})
plt.rcParams.update({"xtick.top": True})
plt.rcParams.update({"ytick.right": True})

from SpectrumMap import *
from MapSeries import *
from DimensionReduction import *
from LinearCombinationFit import *
from make_lorentzian_params import *
from TwoPassFit import *
from percentile_vmax import *