# +
import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen

PRECIPITATIONS = ['Rainfall January', 'Rainfall February','Rainfall March', 'Rainfall April',
                   'Rainfall May','Rainfall June', 'Rainfall July', 'Rainfall August',
                   'Rainfall September', 'Rainfall October', 'Rainfall November', 'Rainfall December']

TMAX = ['Tmax January', 'Tmax February', 'Tmax March', 'Tmax April', 'Tmax May', 'Tmax June',
        'Tmax July', 'Tmax August', 'Tmax September', 'Tmax October', 'Tmax November', 'Tmax December']

TMIN = ['Tmin January', 'Tmin February', 'Tmin March', 'Tmin April', 'Tmin May', 'Tmin June',
        'Tmin July', 'Tmin August', 'Tmin September', 'Tmin October', 'Tmin November', 'Tmin December']

CLIMATE_VARIABLES = PRECIPITATIONS + TMAX + TMIN