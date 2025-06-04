# +
import os
import glob
import pandas as pd
import rasterio
from rasterio.sample import sample_gen

current_variables = {'P_01_Current': 'Rainfall January', 'P_02_Current': 'Rainfall February',
                  'P_03_Current': 'Rainfall March', 'P_04_Current': 'Rainfall April',
                  'P_05_Current': 'Rainfall May', 'P_06_Current': 'Rainfall June',
                  'P_07_Current': 'Rainfall July', 'P_08_Current': 'Rainfall August',
                  'P_09_Current': 'Rainfall September', 'P_10_Current': 'Rainfall October',
                  'P_11_Current': 'Rainfall November', 'P_12_Current': 'Rainfall December',
                  'Tmax_01_Current': 'Tmax January', 'Tmax_02_Current': 'Tmax February',
                  'Tmax_03_Current': 'Tmax March', 'Tmax_04_Current': 'Tmax April',
                  'Tmax_05_Current': 'Tmax May', 'Tmax_06_Current': 'Tmax June',
                  'Tmax_07_Current': 'Tmax July', 'Tmax_08_Current': 'Tmax August',
                  'Tmax_09_Current': 'Tmax September', 'Tmax_10_Current': 'Tmax October',
                  'Tmax_11_Current': 'Tmax November', 'Tmax_12_Current': 'Tmax December',
                  'Tmin_01_Current': 'Tmin January', 'Tmin_02_Current': 'Tmin February',
                  'Tmin_03_Current': 'Tmin March', 'Tmin_04_Current': 'Tmin April',
                  'Tmin_05_Current': 'Tmin May', 'Tmin_06_Current': 'Tmin June',
                  'Tmin_07_Current': 'Tmin July', 'Tmin_08_Current': 'Tmin August',
                  'Tmin_09_Current': 'Tmin September', 'Tmin_10_Current': 'Tmin October',
                  'Tmin_11_Current': 'Tmin November', 'Tmin_12_Current': 'Tmin December'}

ssp245_variables = {'P_01_SSP245': 'Rainfall January', 'P_02_SSP245': 'Rainfall February',
                  'P_03_SSP245': 'Rainfall March', 'P_04_SSP245': 'Rainfall April',
                  'P_05_SSP245': 'Rainfall May', 'P_06_SSP245': 'Rainfall June',
                  'P_07_SSP245': 'Rainfall July', 'P_08_SSP245': 'Rainfall August',
                  'P_09_SSP245': 'Rainfall September', 'P_10_SSP245': 'Rainfall October',
                  'P_11_SSP245': 'Rainfall November', 'P_12_SSP245': 'Rainfall December',
                  'Tmax_01_SSP245': 'Tmax January', 'Tmax_02_SSP245': 'Tmax February',
                  'Tmax_03_SSP245': 'Tmax March', 'Tmax_04_SSP245': 'Tmax April',
                  'Tmax_05_SSP245': 'Tmax May', 'Tmax_06_SSP245': 'Tmax June',
                  'Tmax_07_SSP245': 'Tmax July', 'Tmax_08_SSP245': 'Tmax August',
                  'Tmax_09_SSP245': 'Tmax September', 'Tmax_10_SSP245': 'Tmax October',
                  'Tmax_11_SSP245': 'Tmax November', 'Tmax_12_SSP245': 'Tmax December',
                  'Tmin_01_SSP245': 'Tmin January', 'Tmin_02_SSP245': 'Tmin February',
                  'Tmin_03_SSP245': 'Tmin March', 'Tmin_04_SSP245': 'Tmin April',
                  'Tmin_05_SSP245': 'Tmin May', 'Tmin_06_SSP245': 'Tmin June',
                  'Tmin_07_SSP245': 'Tmin July', 'Tmin_08_SSP245': 'Tmin August',
                  'Tmin_09_SSP245': 'Tmin September', 'Tmin_10_SSP245': 'Tmin October',
                  'Tmin_11_SSP245': 'Tmin November', 'Tmin_12_SSP245': 'Tmin December'}

ssp585_variables = {'P_01_SSP585': 'Rainfall January', 'P_02_SSP585': 'Rainfall February',
                  'P_03_SSP585': 'Rainfall March', 'P_04_SSP585': 'Rainfall April',
                  'P_05_SSP585': 'Rainfall May', 'P_06_SSP585': 'Rainfall June',
                  'P_07_SSP585': 'Rainfall July', 'P_08_SSP585': 'Rainfall August',
                  'P_09_SSP585': 'Rainfall September', 'P_10_SSP585': 'Rainfall October',
                  'P_11_SSP585': 'Rainfall November', 'P_12_SSP585': 'Rainfall December',
                  'Tmax_01_SSP585': 'Tmax January', 'Tmax_02_SSP585': 'Tmax February',
                  'Tmax_03_SSP585': 'Tmax March', 'Tmax_04_SSP585': 'Tmax April',
                  'Tmax_05_SSP585': 'Tmax May', 'Tmax_06_SSP585': 'Tmax June',
                  'Tmax_07_SSP585': 'Tmax July', 'Tmax_08_SSP585': 'Tmax August',
                  'Tmax_09_SSP585': 'Tmax September', 'Tmax_10_SSP585': 'Tmax October',
                  'Tmax_11_SSP585': 'Tmax November', 'Tmax_12_SSP585': 'Tmax December',
                  'Tmin_01_SSP585': 'Tmin January', 'Tmin_02_SSP585': 'Tmin February',
                  'Tmin_03_SSP585': 'Tmin March', 'Tmin_04_SSP585': 'Tmin April',
                  'Tmin_05_SSP585': 'Tmin May', 'Tmin_06_SSP585': 'Tmin June',
                  'Tmin_07_SSP585': 'Tmin July', 'Tmin_08_SSP585': 'Tmin August',
                  'Tmin_09_SSP585': 'Tmin September', 'Tmin_10_SSP585': 'Tmin October',
                  'Tmin_11_SSP585': 'Tmin November', 'Tmin_12_SSP585': 'Tmin December'}
