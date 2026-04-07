import os, io, sys, requests, PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from urllib.error import HTTPError
import math, random, torch
from torch.utils.data import Dataset, DataLoader, json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
elif '' in sys.path:
    sys.path.remove('')

from geotessera import GeoTessera
import ee, cv2, tifffile

downscale = 10

output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

gt = GeoTessera()
print("downloading files")
embedding, crs, transform = gt.fetch_embedding(
    lon=-0.1276,
    lat=51.5072,
    year=2020
)

h, w, c = embedding.shape

new_w = w // downscale ; new_h = h // downscale
embedding = cv2.resize(embedding, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
h, w, c = embedding.shape

# left_lon = transform[2] ; top_lat = transform[5]
# pixel_width = transform[0] ; pixel_height = transform[4] # This is usually negative because pixels go 'down'

# # Boundaries
# right_lon = left_lon + (w * pixel_width)
# bottom_lat = top_lat + (h * pixel_height)

# # The standard bounding box (min_x, min_y, max_x, max_y)
# tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)

# print(f"The exact bounding box for this tile is: {tile_bbox}")

clean_crs = str(crs)
clean_transform = [float(x) for x in transform[:6]]
clean_transform[0] = clean_transform[0] * downscale  # New pixel width
clean_transform[4] = clean_transform[4] * downscale

left_lon = clean_transform[2] ; top_lat = clean_transform[5]
right_lon = left_lon + (new_w * clean_transform[0])
bottom_lat = top_lat + (new_h * clean_transform[4])
tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)
print(f"The exact bounding box for this tile is: {tile_bbox}")


ee.Initialize(project="alexcloud-489214")  # already authenticated
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

# agb_resampled = agb_image.reproject(crs=clean_crs, crsTransform=clean_transform)

url = agb_image.getDownloadURL({
    'region': london_geom,
    'format': 'GEO_TIFF', 
    'dimensions': f"{w}x{h}", # Forces GEE to match GeoTessera's width/height exactly
    'crs': clean_crs,
    'crsTransform': clean_transform,
})

response = requests.get(url)
# with open(file_path, 'wb') as f:
#     for chunk in response.iter_content(chunk_size=8192):
#         f.write(chunk)
with io.BytesIO(response.content) as f:
    # Use PIL to open the TIFF and convert to a numpy array
    agb_array = np.array(PIL.Image.open(f))

agb_array = np.nan_to_num(agb_array, nan=0.0, posinf=0.0, neginf=0.0)
agb_array[agb_array < 0] = 0
# agb_data = agb_resampled.sampleRectangle(region=london_geom).getInfo()
# agb_array = np.array(agb_data['properties']['AGB'])

print(f"GeoTessera Shape: {embedding.shape[:2]}")
print(f"Biomass Array Shape: {agb_array.shape}")


# flat_emb = embedding.reshape(-1, c)
# pca = PCA(n_components=3)
# rgb_emb = pca.fit_transform(flat_emb).reshape(h, w, 3)

# # Normalize RGB to [0, 1] for display
# rgb_emb = (rgb_emb - rgb_emb.min()) / (rgb_emb.max() - rgb_emb.min())

# # 2. Plotting side-by-side
# fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# # Left side: GeoTessera Embedding
# ax[0].imshow(rgb_emb)
# ax[0].set_title("GeoTessera Embedding (PCA-RGB)", fontsize=14)
# ax[0].axis("off")

# # Right side: ESA Biomass
# # Note: we use 'origin=upper' to match standard image indexing
# im2 = ax[1].imshow(agb_array, cmap="YlGn") 
# ax[1].set_title("ESA Biomass CCI (Mg/ha)", fontsize=14)
# ax[1].axis("off")

# # Add a colorbar to the Biomass plot
# plt.colorbar(im2, ax=ax[1], label="Biomass Density", fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()




# 1. Setup Directory Structure
output_dir = "data"
emb_dir = os.path.join(output_dir, "embeddings")
ae_emb_dir = os.path.join(output_dir, "ae_embeddings")
target_dir = os.path.join(output_dir, "targets")

for d in [emb_dir, ae_emb_dir, target_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

gt = GeoTessera()
ee.Initialize(project="alexcloud-489214")


locations = [
    # --- EUROPE ---
    {"name": "london_2020", "lat": 51.5072, "lon": -0.1276, "year": 2020},    # Urban/Suburban mix
    {"name": "fontainebleau_2020", "lat": 48.4047, "lon": 2.7016, "year": 2020}, # Dense French Forest
    {"name": "black_forest_2020", "lat": 48.0116, "lon": 8.1950, "year": 2020},  # Classic German Evergreen
    
    # --- AMERICAS ---
    {"name": "amazon_manaus_2020", "lat": -3.1190, "lon": -60.0217, "year": 2020}, # Tropical Rainforest (High Biomass)
    {"name": "redwood_california_2020", "lat": 41.2132, "lon": -124.0046, "year": 2020}, # Massive Conifers
    {"name": "iowa_farmland_2020", "lat": 41.8780, "lon": -93.0977, "year": 2020}, # Agriculture (Seasonal Biomass)
    
    # --- AFRICA & ASIA ---
    {"name": "congo_basin_2020", "lat": -1.6395, "lon": 18.0000, "year": 2020}, # African Jungle
    {"name": "boreal_siberia_2020", "lat": 61.5240, "lon": 105.3188, "year": 2020}, # Taiga/Boreal Forest
    {"name": "borneo_2020", "lat": 0.0000, "lon": 114.0000, "year": 2020}, # Tropical Indo-Pacific
]
successfully_processed = []

for loc in locations:
    name = loc['name'].lower()
    emb_path = os.path.join(emb_dir, f"{name}_x.npy")
    ae_emb_path = os.path.join(ae_emb_dir, f"{name}_ae.npy")
    target_path = os.path.join(target_dir, f"{name}_y.npy")

    # SKIP IF ALREADY DONE
    # if os.path.exists(emb_path) and os.path.exists(target_path) and os.path.exists(ae_emb_path):
    #     print(f"✅ Skipping {name}: already downloaded.")
    #     successfully_processed.append(loc)
    #     continue

    print(f"🔄 Processing {loc['name']}...")

    try:
        point = ee.Geometry.Point([loc['lon'], loc['lat']]) # 10km buffer = 20km x 20km area (~20x20 raw AlphaEarth pixels)
        buffer_size = 10000  
        geom = point.buffer(buffer_size).bounds()
        
        # Extract BBox for external APIs (min_lon, min_lat, max_lon, max_lat)
        coords = geom.coordinates().get(0).getInfo()
        lons, lats = [c[0] for c in coords], [c[1] for c in coords]
        master_bbox = (min(lons), min(lats), max(lons), max(lats))

        # --- 2. FETCH GEOTESSERA (Matches the Center) ---
        # Note: GeoTessera usually fetches a fixed tile around a point.
        # We fetch it and then resize it to our master grid.
        embedding, crs, transform = gt.fetch_embedding(
            lon=loc['lon'], lat=loc['lat'], year=loc['year']
        )
        # Resize to 256x256 to match the GEE outputs
        embedding_resized = cv2.resize(embedding, (256, 256), interpolation=cv2.INTER_LINEAR)

        # --- 3. FETCH ESA BIOMASS (Aligned to Master Geometry) ---
        agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
                    .filterDate(f"{loc['year']}-01-01", f"{loc['year']}-12-31")
                    .first().select('AGB'))

        agb_url = agb_image.getDownloadURL({
            'region': geom,
            'format': 'GEO_TIFF',
            'dimensions': '256x256', # Force grid size
            'crs': 'EPSG:4326'       # Consistent projection
        })
        
        agb_resp = requests.get(agb_url, timeout=60)
        agb_resp.raise_for_status()
        with io.BytesIO(agb_resp.content) as f:
            agb_array = tifffile.imread(f)
            agb_array = np.nan_to_num(agb_array, nan=0.0).astype(np.float32)
            if agb_array.ndim == 2: agb_array = np.expand_dims(agb_array, axis=-1)

        # --- 4. FETCH ALPHAEARTH (Aligned to Master Geometry) ---
        ae_coll = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filterBounds(geom)
                .sort('system:time_start', False))
        print(ae_coll.size().getInfo())
        ae_img = ae_coll.first().unmask(0).toFloat()
        
        total_bands = 64 ; chunk_size = 8 ; all_ae_chunks = []
        bands_name = [f"A{i:02d}" for i in range(64)]

        for start_band in range(0, total_bands, chunk_size):
            end_band = min(start_band + chunk_size, total_bands)
            # band_names = [f"embedding_{i}" for i in range(start_band, end_band)]
            band_names = bands_name[start_band:end_band]
            print(f"   Fetching AlphaEarth bands: {band_names} for {loc['name']}...")
            ae_chunk_img = ae_img.select(band_names)
            # ae_chunk_img = ae_img.select(list(range(start_band, end_band)))
            
            ae_chunk_url = ae_chunk_img.getDownloadURL({
                'region': geom,
                'format': 'GEO_TIFF',
                'dimensions': '256x256',
                'crs': 'EPSG:4326'
            })
            
            ae_resp = requests.get(ae_chunk_url, timeout=60)
            ae_resp.raise_for_status()
            with io.BytesIO(ae_resp.content) as f:
                chunk_data = tifffile.imread(f)
                chunk_data = np.nan_to_num(chunk_data, nan=0.0)
                if chunk_data.ndim == 2: chunk_data = np.expand_dims(chunk_data, axis=-1)
                all_ae_chunks.append(chunk_data)

        ae_array = np.concatenate(all_ae_chunks, axis=-1)

        # --- 5. SAVE ---
        np.save(emb_path, embedding_resized)
        np.save(target_path, agb_array)
        np.save(ae_emb_path, ae_array)
        
        successfully_processed.append(loc)
        print(f"   ✅ Saved {name}. All arrays are 256x256.")

    except HTTPError as e:
        print(f"   ⚠️ Could not find {name} on GeoTessera server (404).")
    except Exception as e:
        print(f"   ❌ Error processing {name}: {e}")

# --- LOOP 2: VISUALIZE ONLY SUCCESSES ---
if not successfully_processed:
    print("No data available to visualize.")
else:
    print("\nGenerating visualization grid...")
    

    n = len(successfully_processed)
    cols = math.ceil(math.sqrt(n))  # Number of pairs per row
    rows = math.ceil(n / cols)      # Number of rows needed
    
    # Change cols multiplier from 2 to 3
    fig, axes = plt.subplots(rows, 3*cols, figsize=(4*3*cols, 4*rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, loc in enumerate(successfully_processed):
        row = idx // cols
        col = idx % cols
        fname = loc['name'].lower()
        emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
        agb = np.load(os.path.join(target_dir, f"{fname}_y.npy"))
        ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))  # load AE

        # Column 1: Satellite Embedding (PCA)
        h, w, c = emb.shape
        pca_rgb = PCA(n_components=3).fit_transform(emb.reshape(-1, c)).reshape(h, w, 3)
        pca_rgb = (pca_rgb - pca_rgb.min()) / (pca_rgb.max() - pca_rgb.min())
        axes[row, 3*col].imshow(pca_rgb)
        axes[row, 3*col].set_title(f"{loc['name']} Embedding", fontsize=10)
        axes[row, 3*col].axis('off')

        # Column 2: AGB Biomass
        im = axes[row, 3*col+1].imshow(agb, cmap='YlGn')
        axes[row, 3*col+1].set_title(f"{loc['name']} Biomass", fontsize=10)
        axes[row, 3*col+1].axis('off')
        plt.colorbar(im, ax=axes[row, 3*col+1], fraction=0.046, pad=0.04)


        if ae_emb.ndim == 3:
            h2, w2, c2 = ae_emb.shape
            
            # 1. Check for data existence
            if np.all(ae_emb == 0) or np.all(np.isnan(ae_emb)):
                axes[row, 3*col+2].text(0.5, 0.5, "DATA IS ALL ZEROS", ha='center')
            else:
                # 2. Standardize to mean=0, var=1 before PCA
                flat_ae = ae_emb.reshape(-1, c2) ; scaled_ae = StandardScaler().fit_transform(flat_ae)
                ae_pca = PCA(n_components=3).fit_transform(scaled_ae) ; ae_pca = ae_pca.reshape(h2, w2, 3)# 3. PCA
                
                # 4. Robust Normalization (Prevents "All White" images) # We clip extreme outliers (top/bottom 2%)
                p_low, p_high = np.percentile(ae_pca, (2, 98))
                
                # Avoid division by zero if p_high == p_low
                if p_high > p_low:
                    ae_pca = np.clip((ae_pca - p_low) / (p_high - p_low), 0, 1)
                else:
                    ae_pca = np.zeros_like(ae_pca) # Fallback if no variance
                    
                axes[row, 3*col+2].imshow(ae_pca)
            axes[row, 3*col+2].set_title(f"{loc['name']} AE Embed", fontsize=10)
            axes[row, 3*col+2].axis('off')
        

    # Hide unused axes
    for idx in range(n, rows*cols):
        row = idx // cols
        col = idx % cols
        axes[row, 3*col].axis('off')
        axes[row, 3*col+1].axis('off')
        axes[row, 3*col+2].axis('off')

    plt.tight_layout()
    plt.show()






from scipy.stats import pearsonr

def plot_average_correlations(processed_list):
    if not processed_list:
        print("No data to average.")
        return

    # 0. Setup Plots Directory
    plot_dir = os.path.join(output_dir, "correlation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_gt_corrs = []
    all_ae_corrs = []

    print(f"Aggregating correlations for {len(processed_list)} locations...")

    for loc in processed_list:
        fname = loc['name'].lower()
        
        # Load data
        gt_emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
        ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))
        agb = np.load(os.path.join(target_dir, f"{fname}_y.npy")).flatten()

        # Calculate GT correlations for this image
        gt_img_corrs = []
        for c in range(gt_emb.shape[-1]):
            r, _ = pearsonr(gt_emb[:, :, c].flatten(), agb)
            gt_img_corrs.append(np.nan_to_num(r))
        all_gt_corrs.append(gt_img_corrs)

        # Calculate AE correlations for this image
        ae_img_corrs = []
        for c in range(ae_emb.shape[-1]):
            r, _ = pearsonr(ae_emb[:, :, c].flatten(), agb)
            ae_img_corrs.append(np.nan_to_num(r))
        all_ae_corrs.append(ae_img_corrs)

    # 1. Compute Averages
    avg_gt = np.mean(all_gt_corrs, axis=0)
    avg_ae = np.mean(all_ae_corrs, axis=0)
    
    # Optional: Compute Standard Deviation to show consistency
    std_gt = np.std(all_gt_corrs, axis=0)
    std_ae = np.std(all_ae_corrs, axis=0)

    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # GeoTessera Average
    ax1.bar(range(len(avg_gt)), avg_gt, yerr=std_gt, color='skyblue', edgecolor='navy', alpha=0.8, capsize=3)
    ax1.set_title(f"Average GeoTessera vs Biomass Correlation (n={len(processed_list)})", fontsize=14)
    ax1.set_ylabel("Mean Pearson r")
    ax1.set_ylim(-1, 1)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # AlphaEarth Average
    ax2.bar(range(len(avg_ae)), avg_ae, yerr=std_ae, color='salmon', edgecolor='darkred', alpha=0.8, capsize=3)
    ax2.set_title(f"Average AlphaEarth vs Biomass Correlation (n={len(processed_list)})", fontsize=14)
    ax2.set_ylabel("Mean Pearson r")
    ax2.set_xlabel("Channel Index")
    ax2.set_ylim(-1, 1)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # 3. Save
    save_path = os.path.join(plot_dir, "global_average_correlation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Global average correlation plot saved to: {save_path}")
    plt.show()

# Run the averaging plot
plot_average_correlations(successfully_processed)