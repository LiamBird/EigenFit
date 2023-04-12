import sys
try:
    [sys.path.remove(name for name in sys.path if "SpectroscoPy" in name)]
except:
    pass
from SpectrumMap import SpectrumMap
from MapSeries import MapSeries
from DimensionReduction import SeriesDimensionReduction, DimensionReduction
from EigenSelectPeaks import SelectPeaks
from EigenFitResults import FitResults
from CRE_stacks import CRE_stacks
from summary_graphs import *