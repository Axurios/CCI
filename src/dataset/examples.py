import rasterio
import os, io, sys, requests, math
import numpy as np
import matplotlib.pyplot as plt
from urllib.error import HTTPError
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
elif '' in sys.path:
    sys.path.remove('')

from geotessera import GeoTessera
import ee, cv2, tifffile


output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Setup Directory Structure
output_dir = "data"
emb_dir = os.path.join(output_dir, "embeddings")
ae_emb_dir = os.path.join(output_dir, "ae_embeddings")
target_dir = os.path.join(output_dir, "targets")

for d in [emb_dir, ae_emb_dir, target_dir]:
    if not os.path.exists(d):
        os.makedirs(d)


gt = GeoTessera()
ee.Initialize(project="alexcloud-489214")

# handpicked example locations to test (lat, lon, year)
locations = [
    # --- EUROPE ---
    {"name": "london_2020", "lat": 51.5072, "lon": -0.1276, "year": 2020},    # Urban/Suburban mix
    {"name": "fontainebleau_2020", "lat": 48.4047, "lon": 2.7016, "year": 2020}, # Dense French Forest
    {"name": "black_forest_2020", "lat": 48.0116, "lon": 8.1950, "year": 2020},  # Classic German Evergreen
    
    # # --- AMERICAS ---
    # {"name": "amazon_manaus_2020", "lat": -3.1190, "lon": -60.0217, "year": 2020}, # Tropical Rainforest (High Biomass)
    # {"name": "redwood_california_2020", "lat": 41.2132, "lon": -124.0046, "year": 2020}, # Massive Conifers
    # {"name": "iowa_farmland_2020", "lat": 41.8780, "lon": -93.0977, "year": 2020}, # Agriculture (Seasonal Biomass)
    
    # # --- AFRICA & ASIA ---
    # {"name": "congo_basin_2020", "lat": -1.6395, "lon": 18.0000, "year": 2020}, # African Jungle
    # {"name": "boreal_siberia_2020", "lat": 61.5240, "lon": 105.3188, "year": 2020}, # Taiga/Boreal Forest
    # {"name": "borneo_2020", "lat": 0.0000, "lon": 114.0000, "year": 2020}, # Tropical Indo-Pacific
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
        # point = ee.Geometry.Point([loc['lon'], loc['lat']]) # 10km buffer = 20km x 20km area (~20x20 raw AlphaEarth pixels)
        # buffer_size = 10000 ; master_dim = 256 ; crs = 'EPSG:4326'
        # geom = point.buffer(buffer_size).bounds()
        buffer_deg = 0.09  # ~10km at equator
        min_lon = loc['lon'] - buffer_deg ; max_lon = loc['lon'] + buffer_deg
        min_lat = loc['lat'] - buffer_deg ; max_lat = loc['lat'] + buffer_deg

        geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        master_dim = 256 # grid dimension 256x256

        # ---  FETCH ALPHAEARTH (Aligned to Master Geometry) ---
        ae_coll = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filterBounds(geom)
                .sort('system:time_start', False))
        # print(ae_coll.size().getInfo())
        ae_img = ae_coll.first().unmask(0).toFloat()
        
        total_bands = 64 ; chunk_size = 8 ; all_ae_chunks = []
        bands_name = [f"A{i:02d}" for i in range(64)]

        # Download one band just to get geometry info
        first_band = ae_img.select(bands_name[0])
        url = first_band.getDownloadURL({'region': geom,'format': 'GEO_TIFF','dimensions': '256x256', 'crs': 'EPSG:4326'})

        resp = requests.get(url, timeout=60) ; resp.raise_for_status()
        with io.BytesIO(resp.content) as f:
            ae_ref = tifffile.imread(f)
        ae_height, ae_width = ae_ref.shape
        ae_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, ae_width, ae_height)

        for start_band in range(0, total_bands, chunk_size):
            end_band = min(start_band + chunk_size, total_bands)
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
            
            ae_resp = requests.get(ae_chunk_url, timeout=60) ; ae_resp.raise_for_status()
            with io.BytesIO(ae_resp.content) as f:
                chunk_data = tifffile.imread(f)
                chunk_data = np.nan_to_num(chunk_data, nan=0.0)
                if chunk_data.ndim == 2: chunk_data = np.expand_dims(chunk_data, axis=-1)
                all_ae_chunks.append(chunk_data)

        ae_array = np.concatenate(all_ae_chunks, axis=-1)

        # # ---  FETCH TESSERA EMBEDDINGS --- # biggest issue to align
        # bbox = (min_lon, min_lat, max_lon, max_lat)# (min_lon, min_lat, max_lon, max_lat)
        # tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
        # tiles = list(gt.fetch_embeddings(tiles_to_fetch))
        # embedding, tes_crs, tes_transform = gt.fetch_embedding(lon=loc['lon'], lat=loc['lat'], year=loc['year'])
        
      


        # ---  FETCH ESA BIOMASS (Aligned to Master Geometry) ---
        agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
                    .filterDate(f"{loc['year']}-01-01", f"{loc['year']}-12-31")
                    .filterBounds(geom)
                    .first().select('AGB'))

        agb_url = agb_image.getDownloadURL({
            'region': geom,'format': 'GEO_TIFF','dimensions': '256x256','crs': 'EPSG:4326'
        })
        
        agb_resp = requests.get(agb_url, timeout=60)
        agb_resp.raise_for_status()
        with io.BytesIO(agb_resp.content) as f:
            agb_array = tifffile.imread(f)
            agb_array = np.nan_to_num(agb_array, nan=0.0).astype(np.float32)
            if agb_array.ndim == 2: agb_array = np.expand_dims(agb_array, axis=-1)


        # --- SAVE ---
        # np.save(emb_path,embedding) ;
        np.save(target_path, agb_array) ; np.save(ae_emb_path, ae_array)
        
        successfully_processed.append(loc)
        print(f"   ✅ Saved {name}. All arrays are 256x256.")

    except HTTPError as e:
        print(f"   ⚠️ Could not find {name} on GeoTessera server (404).")
    except Exception as e:
        print(f"   ❌ Error processing {name}: {e}")


    
def plot_array(ax, array, title="", cmap=None, pca=False):
    """
    Plots an array on a given axis.
    If pca=True, performs PCA to 3 components and normalizes to [0,1].
    """
    if array.ndim == 3 and pca:
        h, w, c = array.shape
        flat = array.reshape(-1, c)
        flat_scaled = StandardScaler().fit_transform(flat)
        array = PCA(n_components=3).fit_transform(flat_scaled).reshape(h, w, 3)
        # Robust normalization (clip extreme outliers)
        p_low, p_high = np.percentile(array, (2, 98))
        if p_high > p_low:
            array = np.clip((array - p_low) / (p_high - p_low), 0, 1)
        else:
            array = np.zeros_like(array)
    elif array.ndim == 2:
        array = np.nan_to_num(array, nan=0.0)
    
    im = ax.imshow(array, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    return im  # Return im for colorbar if needed


# --- LOOP 2: VISUALIZE ONLY SUCCESSES ---
if not successfully_processed:
    print("No data available to visualize.")
else:
    print("\nGenerating visualization grid...")

    n = len(successfully_processed)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, 2*cols, figsize=(4*2*cols, 4*rows))
    if rows == 1: axes = np.expand_dims(axes, axis=0)
    if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)

    for idx, loc in enumerate(successfully_processed):
        row, col = divmod(idx, cols)
        fname = loc['name'].lower()
        # emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
        agb = np.load(os.path.join(target_dir, f"{fname}_y.npy"))
        ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))

        # Column 1: Satellite Embedding
        # plot_array(axes[row, 3*col], emb, f"{loc['name']} Embedding", pca=True)

        # Column 2: AGB Biomass (with independent colorbar)
        im = axes[row, 2*col].imshow(agb[...,0], cmap='YlGn')
        axes[row, 2*col].set_title(f"{loc['name']} Biomass", fontsize=10)
        axes[row, 2*col].axis('off')
        # Add colorbar without shrinking the axis
        # cbar = fig.colorbar(im, ax=axes[row, 2*col+1], orientation='vertical', fraction=0.05, pad=0.02)
        # cbar.ax.tick_params(labelsize=8)

        # Column 2: AE Embedding
        if ae_emb.ndim == 3 and (not np.all(ae_emb==0) and not np.all(np.isnan(ae_emb))):
            plot_array(axes[row, 2*col+1], ae_emb, f"{loc['name']} AE Embed", pca=True)
        else:
            axes[row, 2*col+1].text(0.5, 0.5, "DATA IS ALL ZEROS", ha='center')
            axes[row, 2*col+1].set_title(f"{loc['name']} AE Embed", fontsize=10)
            axes[row, 2*col+1].axis('off')

    # Hide unused axes
    for idx in range(n, rows*cols):
        row, col = divmod(idx, cols)
        for offset in range(2):
            axes[row, 2*col + offset].axis('off')

    plt.tight_layout()
    plt.show()

























# # --- LOOP 2: VISUALIZE ONLY SUCCESSES ---
# if not successfully_processed:
#     print("No data available to visualize.")
# else:
#     print("\nGenerating visualization grid...")

#     n = len(successfully_processed)
#     cols = math.ceil(math.sqrt(n))
#     rows = math.ceil(n / cols)

#     fig, axes = plt.subplots(rows, 3*cols, figsize=(4*3*cols, 4*rows))
#     if rows == 1: axes = np.expand_dims(axes, axis=0)
#     if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)

#     for idx, loc in enumerate(successfully_processed):
#         row, col = divmod(idx, cols)
#         fname = loc['name'].lower()
#         # emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
#         agb = np.load(os.path.join(target_dir, f"{fname}_y.npy"))
#         ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))

#         # Column 1: Satellite Embedding
#         # plot_array(axes[row, 3*col], emb, f"{loc['name']} Embedding", pca=True)

#         # Column 2: AGB Biomass (with independent colorbar)
#         im = axes[row, 3*col+1].imshow(agb[...,0], cmap='YlGn')
#         axes[row, 3*col+1].set_title(f"{loc['name']} Biomass", fontsize=10)
#         axes[row, 3*col+1].axis('off')
#         # Add colorbar without shrinking the axis
#         # cbar = fig.colorbar(im, ax=axes[row, 3*col+1], orientation='vertical', fraction=0.05, pad=0.02)
#         # cbar.ax.tick_params(labelsize=8)

#         # Column 3: AE Embedding
#         if ae_emb.ndim == 3 and (not np.all(ae_emb==0) and not np.all(np.isnan(ae_emb))):
#             plot_array(axes[row, 3*col+2], ae_emb, f"{loc['name']} AE Embed", pca=True)
#         else:
#             axes[row, 3*col+2].text(0.5, 0.5, "DATA IS ALL ZEROS", ha='center')
#             axes[row, 3*col+2].set_title(f"{loc['name']} AE Embed", fontsize=10)
#             axes[row, 3*col+2].axis('off')

#     # Hide unused axes
#     for idx in range(n, rows*cols):
#         row, col = divmod(idx, cols)
#         for offset in range(3):
#             axes[row, 3*col + offset].axis('off')

#     plt.tight_layout()
#     plt.show()

#        # h, w, c = embedding.shape
#         # left_lon, top_lat = transform[2], transform[5]
#         # pixel_width, pixel_height = transform[0], transform[4]
        
#         # # Calculate BBox
#         # right_lon = left_lon + (w * pixel_width) ; bottom_lat = top_lat + (h * pixel_height)
#         # tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)

#         # # 3. Fetch ESA Biomass from GEE
#         # clean_crs = str(tes_crs) ; clean_transform = [float(x) for x in tes_transform[:6]]
#         # geom = ee.Geometry.Rectangle(coords=list(tile_bbox), proj=clean_crs, geodesic=False)

#         # # Use same bounds as AE/AGB
#         # transform = from_bounds(min_lon, min_lat, max_lon, max_lat, master_dim, master_dim)
#         # height = width = master_dim
#         # count = embedding.shape[2]

#         # # Write GeoTessera embedding to raster and resample to master_dim
#         # with rasterio.MemoryFile() as memfile:
#         #     with memfile.open(
#         #         driver='GTiff',
        #         height=embedding.shape[0],
        #         width=embedding.shape[1],
        #         count=count,
        #         dtype=embedding.dtype,
        #         crs='EPSG:4326',  # same as AE/AGB
        #         transform=transform
        #     ) as dataset:
        #         for i in range(count):
        #             dataset.write(embedding[:, :, i], i+1)

        #         embedding_resized = dataset.read(
        #             out_shape=(count, master_dim, master_dim),
        #             resampling=Resampling.bilinear
        #         )

        # # Swap axes to H x W x C
        # embedding_resized = np.moveaxis(embedding_resized, 0, -1)

# # --- LOOP 2: VISUALIZE ONLY SUCCESSES ---
# if not successfully_processed:
#     print("No data available to visualize.")
# else:
#     print("\nGenerating visualization grid...")

#     n = len(successfully_processed)
#     cols = math.ceil(math.sqrt(n))
#     rows = math.ceil(n / cols)

#     fig, axes = plt.subplots(rows, 3*cols, figsize=(4*3*cols, 4*rows))
#     if rows == 1: axes = np.expand_dims(axes, axis=0)
#     if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)

#     for idx, loc in enumerate(successfully_processed):
#         row, col = divmod(idx, cols)
#         fname = loc['name'].lower()
#         emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
#         agb = np.load(os.path.join(target_dir, f"{fname}_y.npy"))
#         ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))

#         # Column 1: Satellite Embedding
#         plot_array(axes[row, 3*col], emb, f"{loc['name']} Embedding", pca=True)

#         # Column 2: AGB Biomass
#         im = plot_array(axes[row, 3*col+1], agb, f"{loc['name']} Biomass", cmap='YlGn')
#         plt.colorbar(im, ax=axes[row, 3*col+1], fraction=0.046, pad=0.04)

#         # Column 3: AE Embedding
#         if ae_emb.ndim == 3 and (not np.all(ae_emb==0) and not np.all(np.isnan(ae_emb))):
#             plot_array(axes[row, 3*col+2], ae_emb, f"{loc['name']} AE Embed", pca=True)
#         else:
#             axes[row, 3*col+2].text(0.5, 0.5, "DATA IS ALL ZEROS", ha='center')
#             axes[row, 3*col+2].set_title(f"{loc['name']} AE Embed", fontsize=10)
#             axes[row, 3*col+2].axis('off')

#     # Hide unused axes
#     for idx in range(n, rows*cols):
#         row, col = divmod(idx, cols)
#         for offset in range(3):
#             axes[row, 3*col + offset].axis('off')

#     plt.tight_layout()
#     plt.show()





# # --- LOOP 2: VISUALIZE ONLY SUCCESSES ---
# if not successfully_processed:
#     print("No data available to visualize.")
# else:
#     print("\nGenerating visualization grid...")
    

#     n = len(successfully_processed)
#     cols = math.ceil(math.sqrt(n))  # Number of pairs per row
#     rows = math.ceil(n / cols)      # Number of rows needed
    
#     # Change cols multiplier from 2 to 3
#     fig, axes = plt.subplots(rows, 3*cols, figsize=(4*3*cols, 4*rows))

#     if rows == 1:
#         axes = np.expand_dims(axes, axis=0)
#     if axes.ndim == 1:
#         axes = np.expand_dims(axes, axis=0)

#     for idx, loc in enumerate(successfully_processed):
#         row = idx // cols ; col = idx % cols
#         fname = loc['name'].lower()
#         emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
#         agb = np.load(os.path.join(target_dir, f"{fname}_y.npy"))
#         ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))  # load AE

#         # Column 1: Satellite Embedding (PCA)
#         h, w, c = emb.shape
#         pca_rgb = PCA(n_components=3).fit_transform(emb.reshape(-1, c)).reshape(h, w, 3)
#         pca_rgb = (pca_rgb - pca_rgb.min()) / (pca_rgb.max() - pca_rgb.min())
#         axes[row, 3*col].imshow(pca_rgb)
#         axes[row, 3*col].set_title(f"{loc['name']} Embedding", fontsize=10)
#         axes[row, 3*col].axis('off')

#         # Column 2: AGB Biomass
#         im = axes[row, 3*col+1].imshow(agb, cmap='YlGn')
#         axes[row, 3*col+1].set_title(f"{loc['name']} Biomass", fontsize=10)
#         axes[row, 3*col+1].axis('off')
#         plt.colorbar(im, ax=axes[row, 3*col+1], fraction=0.046, pad=0.04)


#         if ae_emb.ndim == 3:
#             h2, w2, c2 = ae_emb.shape
            
#             # 1. Check for data existence
#             if np.all(ae_emb == 0) or np.all(np.isnan(ae_emb)):
#                 axes[row, 3*col+2].text(0.5, 0.5, "DATA IS ALL ZEROS", ha='center')
#             else:
#                 # 2. Standardize to mean=0, var=1 before PCA
#                 flat_ae = ae_emb.reshape(-1, c2) ; scaled_ae = StandardScaler().fit_transform(flat_ae)
#                 ae_pca = PCA(n_components=3).fit_transform(scaled_ae) ; ae_pca = ae_pca.reshape(h2, w2, 3)# 3. PCA
                
#                 # 4. Robust Normalization (Prevents "All White" images) # We clip extreme outliers (top/bottom 2%)
#                 p_low, p_high = np.percentile(ae_pca, (2, 98))
                
#                 # Avoid division by zero if p_high == p_low
#                 if p_high > p_low:
#                     ae_pca = np.clip((ae_pca - p_low) / (p_high - p_low), 0, 1)
#                 else:
#                     ae_pca = np.zeros_like(ae_pca) # Fallback if no variance
                    
#                 axes[row, 3*col+2].imshow(ae_pca)
#             axes[row, 3*col+2].set_title(f"{loc['name']} AE Embed", fontsize=10)
#             axes[row, 3*col+2].axis('off')
        

#     # Hide unused axes
#     for idx in range(n, rows*cols):
#         row = idx // cols ; col = idx % cols
#         axes[row, 3*col].axis('off')
#         axes[row, 3*col+1].axis('off')
#         axes[row, 3*col+2].axis('off')

#     plt.tight_layout()
#     plt.show()






# from scipy.stats import pearsonr

# def plot_average_correlations(processed_list):
#     if not processed_list:
#         print("No data to average.")
#         return

#     # 0. Setup Plots Directory
#     plot_dir = os.path.join(output_dir, "correlation_plots")
#     os.makedirs(plot_dir, exist_ok=True)

#     all_gt_corrs = []
#     all_ae_corrs = []

#     print(f"Aggregating correlations for {len(processed_list)} locations...")

#     for loc in processed_list:
#         fname = loc['name'].lower()
        
#         # Load data
#         gt_emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
#         ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))
#         agb = np.load(os.path.join(target_dir, f"{fname}_y.npy")).flatten()

#         # Calculate GT correlations for this image
#         gt_img_corrs = []
#         for c in range(gt_emb.shape[-1]):
#             r, _ = pearsonr(gt_emb[:, :, c].flatten(), agb)
#             gt_img_corrs.append(np.nan_to_num(r))
#         all_gt_corrs.append(gt_img_corrs)

#         # Calculate AE correlations for this image
#         ae_img_corrs = []
#         for c in range(ae_emb.shape[-1]):
#             r, _ = pearsonr(ae_emb[:, :, c].flatten(), agb)
#             ae_img_corrs.append(np.nan_to_num(r))
#         all_ae_corrs.append(ae_img_corrs)

#     # 1. Compute Averages
#     avg_gt = np.mean(all_gt_corrs, axis=0)
#     avg_ae = np.mean(all_ae_corrs, axis=0)
    
#     # Optional: Compute Standard Deviation to show consistency
#     std_gt = np.std(all_gt_corrs, axis=0)
#     std_ae = np.std(all_ae_corrs, axis=0)

#     # 2. Plotting
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
#     # GeoTessera Average
#     ax1.bar(range(len(avg_gt)), avg_gt, yerr=std_gt, color='skyblue', edgecolor='navy', alpha=0.8, capsize=3)
#     ax1.set_title(f"Average GeoTessera vs Biomass Correlation (n={len(processed_list)})", fontsize=14)
#     ax1.set_ylabel("Mean Pearson r")
#     ax1.set_ylim(-1, 1)
#     ax1.grid(axis='y', linestyle='--', alpha=0.5)

#     # AlphaEarth Average
#     ax2.bar(range(len(avg_ae)), avg_ae, yerr=std_ae, color='salmon', edgecolor='darkred', alpha=0.8, capsize=3)
#     ax2.set_title(f"Average AlphaEarth vs Biomass Correlation (n={len(processed_list)})", fontsize=14)
#     ax2.set_ylabel("Mean Pearson r")
#     ax2.set_xlabel("Channel Index")
#     ax2.set_ylim(-1, 1)
#     ax2.grid(axis='y', linestyle='--', alpha=0.5)

#     plt.tight_layout()
    
#     # 3. Save
#     save_path = os.path.join(plot_dir, "global_average_correlation.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"✅ Global average correlation plot saved to: {save_path}")
#     plt.show()

# # Run the averaging plot
# plot_average_correlations(successfully_processed)




# gt = GeoTessera()
# print("downloading files")
# embedding, crs, transform = gt.fetch_embedding(
#     lon=-0.1276,
#     lat=51.5072,
#     year=2020
# )

# h, w, c = embedding.shape

# new_w = w // downscale ; new_h = h // downscale
# embedding = cv2.resize(embedding, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
# h, w, c = embedding.shape

# # left_lon = transform[2] ; top_lat = transform[5]
# # pixel_width = transform[0] ; pixel_height = transform[4] # This is usually negative because pixels go 'down'

# # # Boundaries
# # right_lon = left_lon + (w * pixel_width)
# # bottom_lat = top_lat + (h * pixel_height)

# # # The standard bounding box (min_x, min_y, max_x, max_y)
# # tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)

# # print(f"The exact bounding box for this tile is: {tile_bbox}")

# clean_crs = str(crs)
# clean_transform = [float(x) for x in transform[:6]]
# clean_transform[0] = clean_transform[0] * downscale  # New pixel width
# clean_transform[4] = clean_transform[4] * downscale

# left_lon = clean_transform[2] ; top_lat = clean_transform[5]
# right_lon = left_lon + (new_w * clean_transform[0])
# bottom_lat = top_lat + (new_h * clean_transform[4])
# tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)
# print(f"The exact bounding box for this tile is: {tile_bbox}")


# ee.Initialize(project="alexcloud-489214")  # already authenticated
# london_geom = ee.Geometry.Rectangle(
#     coords=list(tile_bbox), 
#     proj=clean_crs, 
#     geodesic=False
# )

# agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
#              .filterDate('2020-01-01', '2021-01-01')
#              .first()
#              .select('AGB')
# )

# # agb_resampled = agb_image.reproject(crs=clean_crs, crsTransform=clean_transform)

# url = agb_image.getDownloadURL({
#     'region': london_geom,
#     'format': 'GEO_TIFF', 
#     'dimensions': f"{w}x{h}", # Forces GEE to match GeoTessera's width/height exactly
#     'crs': clean_crs,
#     'crsTransform': clean_transform,
# })

# response = requests.get(url)
# # with open(file_path, 'wb') as f:
# #     for chunk in response.iter_content(chunk_size=8192):
# #         f.write(chunk)
# with io.BytesIO(response.content) as f:
#     # Use PIL to open the TIFF and convert to a numpy array
#     agb_array = np.array(PIL.Image.open(f))

# agb_array = np.nan_to_num(agb_array, nan=0.0, posinf=0.0, neginf=0.0)
# agb_array[agb_array < 0] = 0
# # agb_data = agb_resampled.sampleRectangle(region=london_geom).getInfo()
# # agb_array = np.array(agb_data['properties']['AGB'])

# print(f"GeoTessera Shape: {embedding.shape[:2]}")
# print(f"Biomass Array Shape: {agb_array.shape}")


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











# clean_crs = str(crs)
# clean_transform = [float(x) for x in transform[:6]]

# london_geom = ee.Geometry.Rectangle(
#     coords=list(tile_bbox), 
#     proj=clean_crs, 
#     geodesic=False
# )

# agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
#              .filterDate('2020-01-01', '2021-01-01')
#              .first()
#              .select('AGB')
# )

# agb_resampled = agb_image.reproject(crs=clean_crs, crsTransform=clean_transform)

# url = agb_resampled.getDownloadURL({
#     'region': london_geom,
#     'format': 'GEO_TIFF',
#     'dimensions': f"{w}x{h}" # Forces GEE to match GeoTessera's width/height exactly
# })

# response = requests.get(url)
# # with open(file_path, 'wb') as f:
# #     for chunk in response.iter_content(chunk_size=8192):
# #         f.write(chunk)
# with io.BytesIO(response.content) as f:
#     # Use PIL to open the TIFF and convert to a numpy array
#     agb_array = np.array(PIL.Image.open(f))

# agb_array = np.nan_to_num(agb_array, nan=0.0, posinf=0.0, neginf=0.0)
# agb_array[agb_array < 0] = 0
# # agb_data = agb_resampled.sampleRectangle(region=london_geom).getInfo()
# # agb_array = np.array(agb_data['properties']['AGB'])

# print(f"GeoTessera Shape: {embedding.shape[:2]}")
# print(f"Biomass Array Shape: {agb_array.shape}")


# # flat_emb = embedding.reshape(-1, c)
# # pca = PCA(n_components=3)
# # rgb_emb = pca.fit_transform(flat_emb).reshape(h, w, 3)

# # # Normalize RGB to [0, 1] for display
# # rgb_emb = (rgb_emb - rgb_emb.min()) / (rgb_emb.max() - rgb_emb.min())

# # # 2. Plotting side-by-side
# # fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# # # Left side: GeoTessera Embedding
# # ax[0].imshow(rgb_emb)
# # ax[0].set_title("GeoTessera Embedding (PCA-RGB)", fontsize=14)
# # ax[0].axis("off")

# # # Right side: ESA Biomass
# # # Note: we use 'origin=upper' to match standard image indexing
# # im2 = ax[1].imshow(agb_array, cmap="YlGn") 
# # ax[1].set_title("ESA Biomass CCI (Mg/ha)", fontsize=14)
# # ax[1].axis("off")

# # # Add a colorbar to the Biomass plot
# # plt.colorbar(im2, ax=ax[1], label="Biomass Density", fraction=0.046, pad=0.04)

# # plt.tight_layout()
# # plt.show()




# # 1. Setup Directory Structure
# output_dir = "data"
# emb_dir = os.path.join(output_dir, "embeddings")
# ae_emb_dir = os.path.join(output_dir, "ae_embeddings")
# target_dir = os.path.join(output_dir, "targets")

# for d in [emb_dir, ae_emb_dir, target_dir]:
#     if not os.path.exists(d):
#         os.makedirs(d)

# gt = GeoTessera()
# ee.Initialize(project="alexcloud-489214")


# locations = [
#     # --- EUROPE ---
#     {"name": "london_2020", "lat": 51.5072, "lon": -0.1276, "year": 2020},    # Urban/Suburban mix
#     {"name": "fontainebleau_2020", "lat": 48.4047, "lon": 2.7016, "year": 2020}, # Dense French Forest
#     {"name": "black_forest_2020", "lat": 48.0116, "lon": 8.1950, "year": 2020},  # Classic German Evergreen
    
#     # --- AMERICAS ---
#     {"name": "amazon_manaus_2020", "lat": -3.1190, "lon": -60.0217, "year": 2020}, # Tropical Rainforest (High Biomass)
#     {"name": "redwood_california_2020", "lat": 41.2132, "lon": -124.0046, "year": 2020}, # Massive Conifers
#     {"name": "iowa_farmland_2020", "lat": 41.8780, "lon": -93.0977, "year": 2020}, # Agriculture (Seasonal Biomass)
    
#     # --- AFRICA & ASIA ---
#     {"name": "congo_basin_2020", "lat": -1.6395, "lon": 18.0000, "year": 2020}, # African Jungle
#     {"name": "boreal_siberia_2020", "lat": 61.5240, "lon": 105.3188, "year": 2020}, # Taiga/Boreal Forest
#     {"name": "borneo_2020", "lat": 0.0000, "lon": 114.0000, "year": 2020}, # Tropical Indo-Pacific
# ]
# successfully_processed = []

# for loc in locations:
#     name = loc['name'].lower()
#     emb_path = os.path.join(emb_dir, f"{name}_x.npy")
#     ae_emb_path = os.path.join(ae_emb_dir, f"{name}_ae.npy")
#     target_path = os.path.join(target_dir, f"{name}_y.npy")

#     # SKIP IF ALREADY DONE
#     if os.path.exists(emb_path) and os.path.exists(target_path) and os.path.exists(ae_emb_path):
#         print(f"✅ Skipping {name}: already downloaded.")
#         successfully_processed.append(loc)
#         continue

#     print(f"🔄 Processing {loc['name']}...")

#     try:
#         # 2. Fetch GeoTessera Embedding
#         embedding, crs, transform = gt.fetch_embedding(lon=loc['lon'], lat=loc['lat'], year=loc['year'])
        
#         h, w, c = embedding.shape
#         left_lon, top_lat = transform[2], transform[5]
#         pixel_width, pixel_height = transform[0], transform[4]
        
#         # Calculate BBox
#         right_lon = left_lon + (w * pixel_width)
#         bottom_lat = top_lat + (h * pixel_height)
#         tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)

#         # 3. Fetch ESA Biomass from GEE
#         clean_crs = str(crs)
#         clean_transform = [float(x) for x in transform[:6]]
#         geom = ee.Geometry.Rectangle(coords=list(tile_bbox), proj=clean_crs, geodesic=False)

#         agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
#                     .filterDate(f"{loc['year']}-01-01", f"{loc['year']}-12-31")
#                     .first().select('AGB'))

#         # Reproject to match GeoTessera exactly
#         agb_resampled = agb_image.reproject(crs=clean_crs, crsTransform=clean_transform)
        
#         url = agb_resampled.getDownloadURL({
#             'region': geom,
#             'format': 'GEO_TIFF',
#             'dimensions': f"{w}x{h}"
#         })

#         # 4. Download and Clean
#         response = requests.get(url, timeout=30)
#         response.raise_for_status() # Check for GEE errors
        
#         with io.BytesIO(response.content) as f:
#             agb_array = np.array(PIL.Image.open(f))

#         agb_array = np.nan_to_num(agb_array, nan=0.0, posinf=0.0, neginf=0.0)
#         agb_array[agb_array < 0] = 0


#         # for AlphaEarth Embeddings
#         alpha_earth = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
#                     .filterDate(f"{loc['year']}-01-01", f"{loc['year']}-12-31")
#                     .first())
#         alpha_earth_resampled = alpha_earth.reproject(crs=clean_crs, crsTransform=clean_transform)
#         ae_url = alpha_earth_resampled.getDownloadURL({
#             'region': geom,
#             'format': 'GEO_TIFF',
#             'dimensions': f"{w}x{h}"
#         })

#         # 4. Download and Clean
#         ae_response = requests.get(ae_url, timeout=30)
#         ae_response.raise_for_status() # Check for GEE errors
        
#         with io.BytesIO(ae_response.content) as f:
#             ae_array = np.array(PIL.Image.open(f))

#         ae_array = np.nan_to_num(ae_array, nan=0.0, posinf=0.0, neginf=0.0)
#         ae_array[ae_array < 0] = 0

#         # 5. Save as .npy pairs
#         np.save(emb_path, embedding)
#         np.save(target_path, agb_array)
#         np.save(ae_emb_path, ae_array)
        
#         successfully_processed.append(loc)
#         print(f"   -> Saved {name}. Shape: {embedding.shape[:2]}")

#     except HTTPError as e:
#         print(f"   ⚠️ Could not find {name} on GeoTessera server (404).")
#     except Exception as e:
#         print(f"   ❌ Error processing {name}: {e}")

# # --- LOOP 2: VISUALIZE ONLY SUCCESSES ---
# if not successfully_processed:
#     print("No data available to visualize.")
# else:
#     print("\nGenerating visualization grid...")
    

#     n = len(successfully_processed)
#     cols = math.ceil(math.sqrt(n))  # Number of pairs per row
#     rows = math.ceil(n / cols)      # Number of rows needed
    
#     # Change cols multiplier from 2 to 3
#     fig, axes = plt.subplots(rows, 3*cols, figsize=(4*3*cols, 4*rows))

#     if rows == 1:
#         axes = np.expand_dims(axes, axis=0)
#     if axes.ndim == 1:
#         axes = np.expand_dims(axes, axis=0)

#     for idx, loc in enumerate(successfully_processed):
#         row = idx // cols
#         col = idx % cols
#         fname = loc['name'].lower()
#         emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
#         agb = np.load(os.path.join(target_dir, f"{fname}_y.npy"))
#         ae_emb = np.load(os.path.join(ae_emb_dir, f"{fname}_ae.npy"))  # load AE

#         # Column 1: Satellite Embedding (PCA)
#         h, w, c = emb.shape
#         pca_rgb = PCA(n_components=3).fit_transform(emb.reshape(-1, c)).reshape(h, w, 3)
#         pca_rgb = (pca_rgb - pca_rgb.min()) / (pca_rgb.max() - pca_rgb.min())
#         axes[row, 3*col].imshow(pca_rgb)
#         axes[row, 3*col].set_title(f"{loc['name']} Embedding", fontsize=10)
#         axes[row, 3*col].axis('off')

#         # Column 2: AGB Biomass
#         im = axes[row, 3*col+1].imshow(agb, cmap='YlGn')
#         axes[row, 3*col+1].set_title(f"{loc['name']} Biomass", fontsize=10)
#         axes[row, 3*col+1].axis('off')
#         plt.colorbar(im, ax=axes[row, 3*col+1], fraction=0.046, pad=0.04)

#         # Column 3: AE Embedding (PCA if multi-band, else direct)
#         if ae_emb.ndim == 3:
#             h2, w2, c2 = ae_emb.shape
#             if c2 >= 3:
#                 ae_pca = PCA(n_components=3).fit_transform(ae_emb.reshape(-1, c2)).reshape(h2, w2, 3)
#                 ae_pca = (ae_pca - ae_pca.min()) / (ae_pca.max() - ae_pca.min())
#                 axes[row, 3*col+2].imshow(ae_pca)
#             else:
#                 axes[row, 3*col+2].imshow(ae_emb[:, :, 0], cmap='viridis')
#         else:
#             # Single band
#             axes[row, 3*col+2].imshow(ae_emb, cmap='viridis')
#         axes[row, 3*col+2].set_title(f"{loc['name']} AE Embed", fontsize=10)
#         axes[row, 3*col+2].axis('off')

#     # Hide unused axes
#     for idx in range(n, rows*cols):
#         row = idx // cols
#         col = idx % cols
#         axes[row, 3*col].axis('off')
#         axes[row, 3*col+1].axis('off')
#         axes[row, 3*col+2].axis('off')

#     plt.tight_layout()
#     plt.show()
    # fig, axes = plt.subplots(rows, 2*cols, figsize=(4*2*cols, 4*rows))

    # # If axes is 1D, convert to 2D for consistency
    # if rows == 1:
    #     axes = np.expand_dims(axes, axis=0)
    # if axes.ndim == 1:
    #     axes = np.expand_dims(axes, axis=0)
    
    # for idx, loc in enumerate(successfully_processed):
    #     row = idx // cols
    #     col = idx % cols

    #     fname = loc['name'].lower()
    #     emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
    #     agb = np.load(os.path.join(target_dir, f"{fname}_y.npy"))
        
    #     # PCA for embedding visualization
    #     h, w, c = emb.shape
    #     pca_rgb = PCA(n_components=3).fit_transform(emb.reshape(-1, c)).reshape(h, w, 3)
    #     pca_rgb = (pca_rgb - pca_rgb.min()) / (pca_rgb.max() - pca_rgb.min())

    #     axes[row, 2*col].imshow(pca_rgb)
    #     axes[row, 2*col].set_title(f"{loc['name']} Embedding", fontsize=10)
    #     axes[row, 2*col].axis('off')

    #     im = axes[row, 2*col+1].imshow(agb, cmap='YlGn')
    #     axes[row, 2*col+1].set_title(f"{loc['name']} Biomass", fontsize=10)
    #     axes[row, 2*col+1].axis('off')
    #     plt.colorbar(im, ax=axes[row, 2*col+1], fraction=0.046, pad=0.04)

    # for idx in range(n, rows*cols):
    #     row = idx // cols
    #     col = idx % cols
    #     axes[row, 2*col].axis('off')
    #     axes[row, 2*col+1].axis('off')

    # plt.tight_layout()
    # plt.show()







# import rasterio # Needed to save TIFFs for the visualization functions
# from rasterio.transform import from_origin
# tiff_dir = os.path.join(output_dir, "geotiffs")
# if not os.path.exists(tiff_dir): os.makedirs(tiff_dir)

# for loc in locations:
#     # ... [Your Fetching Logic stays the same] ...
    
#     # 5. Save as GeoTIFF (Required for GeoTessera visualization tools)
#     # We save the first 3 components of the embedding as an RGB TIFF
#     tiff_path = os.path.join(tiff_dir, f"{name}.tif")
    
#     # Create the transform for rasterio
#     # transform = [pixel_width, 0, left_lon, 0, pixel_height, top_lat]
#     res_x, res_y = transform[0], transform[4]
#     west, north = transform[2], transform[5]
#     rio_transform = from_origin(west, north, res_x, abs(res_y))

#     with rasterio.open(
#         tiff_path, 'w', driver='GTiff',
#         height=h, width=w, count=3, # Saving 3 bands (PCA or raw)
#         dtype=embedding.dtype, crs=clean_crs,
#         transform=rio_transform
#     ) as dst:
#         # Write the first 3 channels of the embedding
#         for i in range(3):
#             dst.write(embedding[:, :, i], i + 1)
# print("\n🚀 Running GeoTessera Visualization Suite...")

# # A. Create a Mosaic of all successfully downloaded tiles
# all_tiffs = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tif')]
# if all_tiffs:
#     create_rgb_mosaic(
#         geotiff_paths=all_tiffs,
#         output_path="dataset_mosaic.tif",
#         bands=(0, 1, 2)
#     )
#     print("✅ Mosaic created: dataset_mosaic.tif")

#     # B. Generate Web Tiles (For viewing in a browser/Leaflet)
#     geotiff_to_web_tiles(
#         geotiff_path="dataset_mosaic.tif",
#         output_dir="./web_tiles",
#         zoom_levels=(2, 10)
#     )
#     print("✅ Web tiles generated in ./web_tiles")

# # C. Global Coverage Map
# visualize_global_coverage(
#     tessera_client=gt,
#     output_path="global_coverage.png",
#     year=2020,
#     width_pixels=2000,
#     tile_color="green",
#     tile_alpha=0.5
# )
# print("✅ Global coverage map saved: global_coverage.png")  



# class BiomassDataset(Dataset):
#     def __init__(self, data_dir, patch_size=64, split="train",
#                  split_file=None, use_ae=False, augment=False):
#         self.patch_size = patch_size
#         self.use_ae = use_ae
#         self.augment = augment

#         emb_dir = Path(data_dir) / "embeddings"
#         ae_dir  = Path(data_dir) / "ae_embeddings"
#         tgt_dir = Path(data_dir) / "targets"

#         # Load normalization stats once at init
#         stats_path = Path(data_dir) / "norm_stats.json"
#         if stats_path.exists():
#             with open(stats_path) as f:
#                 stats = json.load(f)
#             self.mean = np.array(stats["mean"], dtype=np.float32)  # (128,)
#             self.std  = np.array(stats["std"],  dtype=np.float32)  # (128,)
#         else:
#             self.mean = None ; self.std  = None

#         # after loading norm_stats.json...
#         ae_stats_path = Path(data_dir) / "norm_stats_ae.json"
#         if ae_stats_path.exists():
#             with open(ae_stats_path) as f:
#                 ae_stats = json.load(f)
#             self.ae_mean = np.array(ae_stats["mean"], dtype=np.float32)
#             self.ae_std  = np.array(ae_stats["std"],  dtype=np.float32)
#         else:
#             self.ae_mean = None ; self.ae_std  = None

#         # Load or create split manifest
#         if split_file and os.path.exists(split_file):
#             with open(split_file) as f:
#                 manifest = json.load(f)
#             names = manifest[split]
#         else:
#             all_names = [f.stem.replace("_x", "") 
#                          for f in emb_dir.glob("*_x.npy")]
#             names = self._default_split(all_names, split)

#         # Memory-map tiles (no RAM cost until sliced)
#         self.tiles = []
#         for name in names:
#             emb = np.load(emb_dir / f"{name}_x.npy", mmap_mode='r')
#             tgt = np.load(tgt_dir / f"{name}_y.npy", mmap_mode='r')
#             ae  = np.load(ae_dir  / f"{name}_ae.npy", mmap_mode='r') \
#                   if use_ae else None

#             H, W, _ = emb.shape
#             n_patches = ((H - patch_size) // patch_size) * \
#                         ((W - patch_size) // patch_size)

#             self.tiles.append({
#                 "emb": emb, "tgt": tgt, "ae": ae,
#                 "H": H, "W": W, "n_patches": n_patches
#             })

#         # Build flat index: (tile_idx, row_start, col_start)
#         self.index = []
#         for ti, tile in enumerate(self.tiles):
#             H, W, P = tile["H"], tile["W"], patch_size
#             for r in range(0, H - P, P):
#                 for c in range(0, W - P, P):
#                     self.index.append((ti, r, c))



#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         ti, r, c = self.index[idx]
#         P = self.patch_size
#         tile = self.tiles[ti]

#         x = tile["emb"][r:r+P, c:c+P, :].copy()  # (P, P, 128)
#         y = tile["tgt"][r:r+P, c:c+P].copy()      # (P, P)

#         # ── Normalize BEFORE concatenation, each embedding independently ──
#         if self.mean is not None:
#             x = (x - self.mean) / (self.std + 1e-6)  # mean is (128,) ✅

#         if self.use_ae and tile["ae"] is not None:
#             ae = tile["ae"][r:r+P, c:c+P, :].copy()
#             # AE has its own scale — normalize it too if stats exist
#             if self.ae_mean is not None:
#                 ae = (ae - self.ae_mean) / (self.ae_std + 1e-6)
#             x = np.concatenate([x, ae], axis=-1)      # (P, P, 128+C_ae)

#         if self.augment:
#             x, y = self._augment(x, y)

#         x = torch.from_numpy(x).permute(2, 0, 1).float()
#         y = torch.from_numpy(y).float()
#         return x, y

#     def _augment(self, x, y):
#         k = np.random.randint(0, 4)
#         x = np.rot90(x, k).copy()
#         y = np.rot90(y, k).copy()
#         if np.random.rand() > 0.5:
#             x = np.fliplr(x).copy()
#             y = np.fliplr(y).copy()
#         return x, y

#     @staticmethod
#     def _default_split(names, split, seed=42):
#         rng = np.random.default_rng(seed)
#         shuffled = rng.permutation(names).tolist()
#         n = len(shuffled)
#         cuts = (int(0.7*n), int(0.85*n))
#         return {"train": shuffled[:cuts[0]],
#                 "val":   shuffled[cuts[0]:cuts[1]],
#                 "test":  shuffled[cuts[1]:]}[split]


# def compute_normalization_stats(data_dir, sample_tiles=20):
#     emb_dir = Path(data_dir) / "embeddings"
#     paths = list(emb_dir.glob("*_x.npy"))[:sample_tiles]

#     count = 0
#     mean = None
#     M2   = None

#     for p in paths:
#         tile = np.load(p).reshape(-1, 128).astype(np.float64)
#         for row in tile:
#             count += 1
#             if mean is None:
#                 mean = row.copy()
#                 M2   = np.zeros(128)
#             else:
#                 delta = row - mean
#                 mean += delta / count
#                 M2   += delta * (row - mean)  # uses updated mean — correct

#     std = np.sqrt(M2 / count)
#     stats = {"mean": mean.tolist(), "std": std.tolist()}

#     with open(Path(data_dir) / "norm_stats.json", "w") as f:
#         json.dump(stats, f)

#     return stats




# print("downloading files")
# embedding, crs, transform = gt.fetch_embedding(
#     lon=-0.1276,
#     lat=51.5072,
#     year=2020
# )

# h, w, c = embedding.shape
# # transform = [pixel_width, 0, left_lon, 0, pixel_height, top_lat]
# left_lon = transform[2] ; top_lat = transform[5]
# pixel_width = transform[0] ; pixel_height = transform[4] # This is usually negative because pixels go 'down'

# # Boundaries
# right_lon = left_lon + (w * pixel_width)
# bottom_lat = top_lat + (h * pixel_height)

# # The standard bounding box (min_x, min_y, max_x, max_y)
# tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)

# print(f"The exact bounding box for this tile is: {tile_bbox}")


