# +
import tifffile
import os

# Path to your input multi-layer TIFF file
#input_path = 'tmax_ssp585_france.tif'
#output_dir = 'Tmax'
input_path = 'tmin_ssp585_france.tif'
output_dir = 'Tmin'

os.makedirs(output_dir, exist_ok=True)

# Read the full array
data = tifffile.imread(input_path)  # shape: (1170, 1764, 12)

# Loop through the 12 layers (last dimension)
num_layers = data.shape[2]
for i in range(num_layers):
    layer = data[:, :, i]
    output_path = os.path.join(output_dir, f'layer_{i+1}.tif')
    tifffile.imwrite(output_path, layer)

print(f"Exported {num_layers} layers to '{output_dir}'")

# -


