# +
from constants import *
import geopandas as gpd
from shapely.geometry import Point

class Extractor:
    """
    This auxiliary class aims to extract climate data given a set of locations.
    """
    
    def __init__(self, input_file, climate_folder, id_stations_name=None):
        """
        Extracts climate data from a raster climate folder at stations' locations.

        Attributes :
        input_file (str): Name of the file with stations locations
        climate_folder (str): Folder with raster data stored in tif files
        id_stations_name (str): Column in the input dataframe with stations ids
        """
        
        self.input_file = input_file
        self.climate_folder = climate_folder
        self.id_stations_name = id_stations_name
        
    def extract(self, climate_variables=CLIMATE_VARIABLES, verbose=True):
        """
        Creates and returns the extracted_data dataframe.
        """

        id_stations_name = self.id_stations_name
        
        df = pd.read_csv('Data/' + self.input_file, sep=",", on_bad_lines='skip')
        
        filter_ = ['lon', 'lat']
        if id_stations_name is not None:
            filter_.append(id_stations_name)
            df = df.drop_duplicates(subset=id_stations_name, keep='first') # Dropping multiple records at one location
            
        df = df[filter_].dropna() # Keeping only locations and possible indices of stations
        coordinates = list(zip(df['lon'], df['lat']))
        
        tif_files = glob.glob(os.path.join(self.climate_folder, "*.tif"))
        m = len(tif_files) # Number of climate variables
        
        # Starting to extract climate data
        for i, tif_file in enumerate(tif_files):
            if verbose:
                print("Extraction: " + str(int(100 * i / m)) + " %",end='\r')
            with rasterio.open(tif_file) as src:
                var_name = os.path.splitext(os.path.basename(tif_file))[0]
                
                if var_name in climate_variables: # Considering valid tif files
                    scale = src.scales[0] if src.scales else 1.0 # Sometimes, temperature data is multiplied by 100
                    offset = src.offsets[0] if src.offsets else 0.0
                    values = [val[0] * scale + offset for val in src.sample(coordinates)]
                    df[var_name] = values
                    df[var_name] = df[var_name].astype(float).round(2)

        df = df[df['Rainfall January'].notna() & (df['Rainfall January'] >= 1)] # Dropping points out of France
        
        if id_stations_name is not None: # Ensuring that ids are integers and not float values
            df[id_stations_name] = df[id_stations_name].astype(int)
            
        df = df.dropna()
        
        if verbose:
            print("Extraction completed")
            print(df.head())
        
        return df
    
    def export(self, extracted_data, output_file):
        """
        Exports climate data as a csv file.
        """
        
        extracted_data.to_csv('Data/' + output_file, index=False)
    
    def filter_france(df):
        """
        Outdated
        """
        
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        gdf_france = gpd.read_file("Data/france_contour/France.shp")

        if gdf_france.crs != gdf_points.crs:
            gdf_france = gdf_france.to_crs(gdf_points.crs)

        return gdf_points[gdf_points.within(gdf_france.unary_union)]
            
    
