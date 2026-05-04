import rasterio
import os, io, sys, requests, math
import numpy as np
import matplotlib.pyplot as plt
from urllib.error import HTTPError
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.transform import from_bounds


if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
elif '' in sys.path:
    sys.path.remove('')

import ee, cv2, tifffile

output_dir = "data_uniform"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Setup Directory Structure
# emb_dir = os.path.join(output_dir, "embeddings")
ae_dir = os.path.join(output_dir, "ae_embeddings")
y_dir = os.path.join(output_dir, "targets")

# for d in [emb_dir, ae_emb_dir, target_dir]:
#     if not os.path.exists(d):
#         os.makedirs(d)


# gt = GeoTessera()
ee.Initialize(project="alexcloud-489214")

successfully_processed = []

def parse_name(filename):
    # "london_2020_y.npy" -> "london_2020"
    return filename.replace("_y.npy", "").replace("_ae.npy", "")



ae_files = sorted([f for f in os.listdir(ae_dir) if f.endswith("_ae.npy")])
for ae_file in ae_files:
    name = parse_name(ae_file)

    ae_path = os.path.join(ae_dir, ae_file)
    y_path  = os.path.join(y_dir, f"{name}_y.npy")

    if not os.path.exists(y_path):
        print(f"⚠️ Missing target for {name}")
        continue

    ae = np.load(ae_path) ; y  = np.load(y_path)

    successfully_processed.append({"name": name,"ae": ae,"y": y})

    print(f"✅ Loaded {name} | AE: {ae.shape} | Y: {y.shape}")

# for loc in locations:
#     name = loc['name'].lower()
#     emb_path = os.path.join(emb_dir, f"{name}_x.npy")
#     ae_emb_path = os.path.join(ae_emb_dir, f"{name}_ae.npy")
#     target_path = os.path.join(target_dir, f"{name}_y.npy")

#     # SKIP IF ALREADY DONE
#     # if os.path.exists(emb_path) and os.path.exists(target_path) and os.path.exists(ae_emb_path):
#     #     print(f"✅ Skipping {name}: already downloaded.")
#     #     successfully_processed.append(loc)
#     #     continue

#     print(f"🔄 Processing {loc['name']}...")

#     try:
#         # point = ee.Geometry.Point([loc['lon'], loc['lat']]) # 10km buffer = 20km x 20km area (~20x20 raw AlphaEarth pixels)
#         # buffer_size = 10000 ; master_dim = 256 ; crs = 'EPSG:4326'
#         # geom = point.buffer(buffer_size).bounds()
#         buffer_deg = 0.09  # ~10km at equator
#         min_lon = loc['lon'] - buffer_deg ; max_lon = loc['lon'] + buffer_deg
#         min_lat = loc['lat'] - buffer_deg ; max_lat = loc['lat'] + buffer_deg

#         geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
#         master_dim = 256 # grid dimension 256x256

#         # ---  FETCH ALPHAEARTH (Aligned to Master Geometry) ---
#         ae_coll = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
#                 .filterBounds(geom)
#                 .sort('system:time_start', False))
#         # print(ae_coll.size().getInfo())
#         ae_img = ae_coll.first().unmask(0).toFloat()
        
#         total_bands = 64 ; chunk_size = 8 ; all_ae_chunks = []
#         bands_name = [f"A{i:02d}" for i in range(64)]

#         # Download one band just to get geometry info
#         first_band = ae_img.select(bands_name[0])
#         url = first_band.getDownloadURL({'region': geom,'format': 'GEO_TIFF','dimensions': '256x256', 'crs': 'EPSG:4326'})

#         resp = requests.get(url, timeout=60) ; resp.raise_for_status()
#         with io.BytesIO(resp.content) as f:
#             ae_ref = tifffile.imread(f)
#         ae_height, ae_width = ae_ref.shape
#         ae_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, ae_width, ae_height)

#         for start_band in range(0, total_bands, chunk_size):
#             end_band = min(start_band + chunk_size, total_bands)
#             band_names = bands_name[start_band:end_band]
#             print(f"   Fetching AlphaEarth bands: {band_names} for {loc['name']}...")
#             ae_chunk_img = ae_img.select(band_names)
#             # ae_chunk_img = ae_img.select(list(range(start_band, end_band)))
            
#             ae_chunk_url = ae_chunk_img.getDownloadURL({
#                 'region': geom,
#                 'format': 'GEO_TIFF',
#                 'dimensions': '256x256',
#                 'crs': 'EPSG:4326'
#             })
            
#             ae_resp = requests.get(ae_chunk_url, timeout=60) ; ae_resp.raise_for_status()
#             with io.BytesIO(ae_resp.content) as f:
#                 chunk_data = tifffile.imread(f)
#                 chunk_data = np.nan_to_num(chunk_data, nan=0.0)
#                 if chunk_data.ndim == 2: chunk_data = np.expand_dims(chunk_data, axis=-1)
#                 all_ae_chunks.append(chunk_data)

#         ae_array = np.concatenate(all_ae_chunks, axis=-1)

#         # # ---  FETCH TESSERA EMBEDDINGS --- # biggest issue to align
#         # bbox = (min_lon, min_lat, max_lon, max_lat)# (min_lon, min_lat, max_lon, max_lat)
#         # tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
#         # tiles = list(gt.fetch_embeddings(tiles_to_fetch))
#         # embedding, tes_crs, tes_transform = gt.fetch_embedding(lon=loc['lon'], lat=loc['lat'], year=loc['year'])
        
      


#         # ---  FETCH ESA BIOMASS (Aligned to Master Geometry) ---
#         agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
#                     .filterDate(f"{loc['year']}-01-01", f"{loc['year']}-12-31")
#                     .filterBounds(geom)
#                     .first().select('AGB'))

#         agb_url = agb_image.getDownloadURL({
#             'region': geom,'format': 'GEO_TIFF','dimensions': '256x256','crs': 'EPSG:4326'
#         })
        
#         agb_resp = requests.get(agb_url, timeout=60)
#         agb_resp.raise_for_status()
#         with io.BytesIO(agb_resp.content) as f:
#             agb_array = tifffile.imread(f)
#             agb_array = np.nan_to_num(agb_array, nan=0.0).astype(np.float32)
#             if agb_array.ndim == 2: agb_array = np.expand_dims(agb_array, axis=-1)


#         # --- SAVE ---
#         # np.save(emb_path,embedding) ;
#         np.save(target_path, agb_array) ; np.save(ae_emb_path, ae_array)
        
#         successfully_processed.append(loc)
#         print(f"   ✅ Saved {name}. All arrays are 256x256.")

#     except HTTPError as e:
#         print(f"   ⚠️ Could not find {name} on GeoTessera server (404).")
#     except Exception as e:
#         print(f"   ❌ Error processing {name}: {e}")









    
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
        # row, big_col = divmod(idx, cols)
        # col = big_col
        row = idx // cols
        col = idx % cols
        fname = loc['name'].lower()
        # emb = np.load(os.path.join(emb_dir, f"{fname}_x.npy"))
        agb = np.load(os.path.join(y_dir, f"{fname}_y.npy"))
        ae_emb = np.load(os.path.join(ae_dir, f"{fname}_ae.npy"))

        # Column 1: Satellite Embedding
        # plot_array(axes[row, 3*col], emb, f"{loc['name']} Embedding", pca=True)

        # Column 2: AGB Biomass (with independent colorbar)
        im = axes[row, 2*col].imshow(agb[...,0], cmap='YlGn')
       #  axes[row, 2*col].set_title(f"{loc['name']} Biomass", fontsize=10)
        axes[row, 2*col].axis('off')
        # Add colorbar without shrinking the axis
        # cbar = fig.colorbar(im, ax=axes[row, 2*col+1], orientation='vertical', fraction=0.05, pad=0.02)
     

        # Column 2: AE Embedding
        if ae_emb.ndim == 3 and (not np.all(ae_emb==0) and not np.all(np.isnan(ae_emb))):
            plot_array(axes[row, 2*col+1], ae_emb, pca=True)
        else:
            axes[row, 2*col+1].text(0.5, 0.5, "DATA IS ALL ZEROS", ha='center')
            # axes[row, 2*col+1].set_title(f"{loc['name']} AE Embed", fontsize=10)
            axes[row, 2*col+1].axis('off')

    # Hide unused axes
    for idx in range(n, rows*cols):
        row, col = divmod(idx, cols)
        for offset in range(2):
            axes[row, 2*col + offset].axis('off')

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.001, hspace=0.001)
    plt.show()


