# +
import os
import glob
import pandas as pd
import rasterio
from rasterio.sample import sample_gen

dict_variables = {'P_01_Current': 'Rainfall January', 'P_02_Current': 'Rainfall February',
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

class Extractor:
    """
    This class aims to extract climate data given a set of points.
    """
    
    def __init__(self, input_file, output_file, id_stations_name=None, id_species_name=None):
        self.input_file = input_file
        self.output_file = output_file
        self.id_stations_name = id_stations_name
        self.id_species_name = id_species_name
        
    def extract(self, verbose=True):
        
        id_species_name = self.id_species_name
        id_stations_name = self.id_stations_name
        
        df = pd.read_csv(self.input_file, sep=",", on_bad_lines='skip')
        
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
        
        tif_files = glob.glob(os.path.join("Current/", "*.tif"))
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
        
        df = df[df.drop(columns=filter_).values.sum(axis=1) != 0] # Dropping points out of France
        
        if id_species_name:
            df[id_species_name] = df[id_species_name].astype(int)
        if id_stations_name:
            df[id_stations_name] = df[id_stations_name].astype(int)
        df = df.dropna()
        
        if verbose:
            print("Extraction completed")
            print(df.head())
            
        df.to_csv(self.output_file, index=False)
