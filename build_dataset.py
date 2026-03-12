import os, io, sys, requests, PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
elif '' in sys.path:
    sys.path.remove('')

from geotessera import GeoTessera
import ee

gt = GeoTessera()

embedding, crs, transform = gt.fetch_embedding(
    lon=-0.1276,
    lat=51.5072,
    year=2020
)

h, w, c = embedding.shape
# transform = [pixel_width, 0, left_lon, 0, pixel_height, top_lat]
left_lon = transform[2] ; top_lat = transform[5]
pixel_width = transform[0] ; pixel_height = transform[4] # This is usually negative because pixels go 'down'

# Boundaries
right_lon = left_lon + (w * pixel_width)
bottom_lat = top_lat + (h * pixel_height)

# The standard bounding box (min_x, min_y, max_x, max_y)
tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)

print(f"The exact bounding box for this tile is: {tile_bbox}")





ee.Initialize(project="alexcloud-489214")  # already authenticated

clean_crs = str(crs)
clean_transform = [float(x) for x in transform[:6]]

london_geom = ee.Geometry.Rectangle(
    coords=list(tile_bbox), 
    proj=clean_crs, 
    geodesic=False
)

agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
             .filterDate('2020-01-01', '2021-01-01')
             .first()
             .select('AGB')
)

agb_resampled = agb_image.reproject(crs=clean_crs, crsTransform=clean_transform)

url = agb_resampled.getDownloadURL({
    'region': london_geom,
    'format': 'GEO_TIFF',
    'dimensions': f"{w}x{h}" # Forces GEE to match GeoTessera's width/height exactly
})

response = requests.get(url)
with io.BytesIO(response.content) as f:
    # Use PIL to open the TIFF and convert to a numpy array
    agb_array = np.array(PIL.Image.open(f))

agb_array = np.nan_to_num(agb_array, nan=0.0, posinf=0.0, neginf=0.0)
agb_array[agb_array < 0] = 0
# agb_data = agb_resampled.sampleRectangle(region=london_geom).getInfo()
# agb_array = np.array(agb_data['properties']['AGB'])

print(f"GeoTessera Shape: {embedding.shape[:2]}")
print(f"Biomass Array Shape: {agb_array.shape}")


flat_emb = embedding.reshape(-1, c)
pca = PCA(n_components=3)
rgb_emb = pca.fit_transform(flat_emb).reshape(h, w, 3)

# Normalize RGB to [0, 1] for display
rgb_emb = (rgb_emb - rgb_emb.min()) / (rgb_emb.max() - rgb_emb.min())

# 2. Plotting side-by-side
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Left side: GeoTessera Embedding
ax[0].imshow(rgb_emb)
ax[0].set_title("GeoTessera Embedding (PCA-RGB)", fontsize=14)
ax[0].axis("off")

# Right side: ESA Biomass
# Note: we use 'origin=upper' to match standard image indexing
im2 = ax[1].imshow(agb_array, cmap="YlGn") 
ax[1].set_title("ESA Biomass CCI (Mg/ha)", fontsize=14)
ax[1].axis("off")

# Add a colorbar to the Biomass plot
plt.colorbar(im2, ax=ax[1], label="Biomass Density", fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()