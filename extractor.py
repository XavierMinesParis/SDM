# +
import os
import glob
import pandas as pd
import rasterio
from rasterio.sample import sample_gen
from constants import *

class Extractor:
    """
    This class aims to extract climate data given a set of points.
    """
    
    def __init__(self, input_file, output_file, id_stations_name=None, id_species_name=None, scenario='current'):
        self.input_file = input_file
        self.output_file = output_file
        self.id_stations_name = id_stations_name
        self.id_species_name = id_species_name
        self.scenario = scenario
        
    def extract(self, verbose=True):
        
        if self.scenario == 'ssp245':
            dict_variables = ssp245_variables
            folder = "SSP245"
        elif self.scenario == 'ssp585':
            dict_variables = ssp585_variables
            folder = "SSP585/"
        else:
            dict_variables = current_variables
            folder = "Current/"
        
        id_species_name = self.id_species_name
        id_stations_name = self.id_stations_name
        
        df = pd.read_csv('Data/' + self.input_file, sep=",", on_bad_lines='skip')
        
        filter_ = ['lon', 'lat']
        if id_species_name is not None:
            filter_.append(id_species_name)
        if id_stations_name is not None:
            filter_.append(id_stations_name)
            
        if id_species_name is not None and id_stations_name is not None:
            df = df.drop_duplicates(subset=[id_stations_name,
                                            id_species_name], keep='first') # Droping multiple records at one location
        elif id_stations_name is not None:
            df = df.drop_duplicates(subset=id_stations_name, keep='first')
            
        df = df[filter_].dropna()    
        coordinates = list(zip(df['lon'], df['lat']))
        
        tif_files = glob.glob(os.path.join(folder, "*.tif"))
        n = len(tif_files)
        
        for i, tif_file in enumerate(tif_files):
            if verbose:
                print("Extraction: " + str(int(100 * i / n)) + " %",end='\r')
            with rasterio.open(tif_file) as src:
                var_name = os.path.splitext(os.path.basename(tif_file))[0]
                
                if var_name in dict_variables.keys():
                    scale = src.scales[0] if src.scales else 1.0
                    offset = src.offsets[0] if src.offsets else 0.0
                    values = [val[0] * scale + offset for val in src.sample(coordinates)]
                    df[dict_variables[var_name]] = values
                    df[dict_variables[var_name]] = df[dict_variables[var_name]].astype(float).round(2)
        print(df)
        df = df[df['Rainfall January'].notna() & (df['Rainfall January'] >= 1)] # Dropping points out of France
        
        if id_species_name:
            df[id_species_name] = df[id_species_name].astype(int)
        if id_stations_name:
            df[id_stations_name] = df[id_stations_name].astype(int)
        df = df.dropna()
        
        if verbose:
            print("Extraction completed")
            print(df.head())
            
        df.to_csv('Data/' + self.output_file, index=False)


# -

"""
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
import os

# Path to reference file (with correct georeference)
reference_path = 'P_01_SSP585.tif'

# Folder containing unreferenced TIFFs
target_folder = 'SSP585'

# Read CRS and transform from reference
with rasterio.open(reference_path) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = ref.shape

# Loop through TIFF files in the folder
for filename in os.listdir(target_folder):
    if not filename.lower().endswith('.tif'):
        continue

    file_path = os.path.join(target_folder, filename)

    # Skip reference file
    if os.path.abspath(file_path) == os.path.abspath(reference_path):
        continue

    with rasterio.open(file_path) as src:
        if src.shape != ref_shape:
            print(f"Skipping {filename} due to shape mismatch: {src.shape} vs {ref_shape}")
            continue

        profile = src.profile.copy()
        profile.update({
            'crs': ref_crs,
            'transform': ref_transform
        })
        data = src.read()

    # Overwrite the original file
    with rasterio.open(file_path, 'w', **profile) as dst:
        dst.write(data)

    print(f"Overwritten with georeference: {file_path}")
"""


