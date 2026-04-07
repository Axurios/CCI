import os, io, sys, requests, PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from urllib.error import HTTPError
import math
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np, os, json
from pathlib import Path

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
elif '' in sys.path:
    sys.path.remove('')

from geotessera import GeoTessera
# from geotessera.visualization import (
#     create_rgb_mosaic,
#     visualize_global_coverage
# )
# from geotessera.web import (
#     create_coverage_summary_map,
#     geotiff_to_web_tiles
# )

import ee

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
    if os.path.exists(emb_path) and os.path.exists(target_path) and os.path.exists(ae_emb_path):
        print(f"✅ Skipping {name}: already downloaded.")
        successfully_processed.append(loc)
        continue

    print(f"🔄 Processing {loc['name']}...")

    try:
        # 2. Fetch GeoTessera Embedding
        embedding, crs, transform = gt.fetch_embedding(lon=loc['lon'], lat=loc['lat'], year=loc['year'])
        
        h, w, c = embedding.shape
        left_lon, top_lat = transform[2], transform[5]
        pixel_width, pixel_height = transform[0], transform[4]
        
        # Calculate BBox
        right_lon = left_lon + (w * pixel_width)
        bottom_lat = top_lat + (h * pixel_height)
        tile_bbox = (left_lon, bottom_lat, right_lon, top_lat)

        # 3. Fetch ESA Biomass from GEE
        clean_crs = str(crs)
        clean_transform = [float(x) for x in transform[:6]]
        geom = ee.Geometry.Rectangle(coords=list(tile_bbox), proj=clean_crs, geodesic=False)

        agb_image = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
                    .filterDate(f"{loc['year']}-01-01", f"{loc['year']}-12-31")
                    .first().select('AGB'))

        # Reproject to match GeoTessera exactly
        agb_resampled = agb_image.reproject(crs=clean_crs, crsTransform=clean_transform)
        
        url = agb_resampled.getDownloadURL({
            'region': geom,
            'format': 'GEO_TIFF',
            'dimensions': f"{w}x{h}"
        })

        # 4. Download and Clean
        response = requests.get(url, timeout=30)
        response.raise_for_status() # Check for GEE errors
        
        with io.BytesIO(response.content) as f:
            agb_array = np.array(PIL.Image.open(f))

        agb_array = np.nan_to_num(agb_array, nan=0.0, posinf=0.0, neginf=0.0)
        agb_array[agb_array < 0] = 0


        # for AlphaEarth Embeddings
        alpha_earth = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                    .filterDate(f"{loc['year']}-01-01", f"{loc['year']}-12-31")
                    .first())
        alpha_earth_resampled = alpha_earth.reproject(crs=clean_crs, crsTransform=clean_transform)
        ae_url = alpha_earth_resampled.getDownloadURL({
            'region': geom,
            'format': 'GEO_TIFF',
            'dimensions': f"{w}x{h}"
        })

        # 4. Download and Clean
        ae_response = requests.get(ae_url, timeout=30)
        ae_response.raise_for_status() # Check for GEE errors
        
        with io.BytesIO(ae_response.content) as f:
            ae_array = np.array(PIL.Image.open(f))

        ae_array = np.nan_to_num(ae_array, nan=0.0, posinf=0.0, neginf=0.0)
        ae_array[ae_array < 0] = 0

        # 5. Save as .npy pairs
        np.save(emb_path, embedding)
        np.save(target_path, agb_array)
        np.save(ae_emb_path, ae_array)
        
        successfully_processed.append(loc)
        print(f"   -> Saved {name}. Shape: {embedding.shape[:2]}")

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

        # Column 3: AE Embedding (PCA if multi-band, else direct)
        if ae_emb.ndim == 3:
            h2, w2, c2 = ae_emb.shape
            if c2 >= 3:
                ae_pca = PCA(n_components=3).fit_transform(ae_emb.reshape(-1, c2)).reshape(h2, w2, 3)
                ae_pca = (ae_pca - ae_pca.min()) / (ae_pca.max() - ae_pca.min())
                axes[row, 3*col+2].imshow(ae_pca)
            else:
                axes[row, 3*col+2].imshow(ae_emb[:, :, 0], cmap='viridis')
        else:
            # Single band
            axes[row, 3*col+2].imshow(ae_emb, cmap='viridis')
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


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────
def compute_normalization_stats(data_dir, sample_tiles=20,
                                subdir="embeddings", out="norm_stats.json"):
    """
    Welford online mean/std over a sample of tiles.
    Works for any embedding dimensionality — infers n_channels from first file.
    """
    emb_dir = Path(data_dir) / subdir
    pattern = "*_x.npy" if subdir == "embeddings" else "*_ae.npy"

    all_paths = list(emb_dir.glob(pattern))
    if not all_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {emb_dir}")

    # 2. Randomly sample 'sample_tiles' or the total count, whichever is smaller
    n_to_sample = min(len(all_paths), sample_tiles) if sample_tiles else len(all_paths)
    paths = random.sample(all_paths, k=n_to_sample)
    # paths = list(emb_dir.glob(pattern))[:sample_tiles]

    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {emb_dir}")

    n_channels = np.load(paths[0]).shape[-1]  # infer instead of hardcoding
    count = 0 ; mean = None ; M2 = None

    for p in paths:
        tile = np.load(p).reshape(-1, n_channels).astype(np.float64)
        for row in tile:
            count += 1
            if mean is None:
                mean = row.copy()
                M2   = np.zeros(n_channels)
            else:
                delta = row - mean
                mean += delta / count
                M2   += delta * (row - mean)  # uses updated mean — correct

    std = np.sqrt(M2 / count)
    stats = {"mean": mean.tolist(), "std": std.tolist()}

    out_path = Path(data_dir) / out
    with open(out_path, "w") as f:
        json.dump(stats, f)

    print(f"Saved normalization stats ({n_channels}ch, {count} pixels) → {out_path}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BiomassDataset(Dataset):
    """
    Streams (patch_embedding, patch_target) pairs from pre-saved .npy tiles.

    Directory layout expected:
        data_dir/
            embeddings/       <name>_x.npy       (H, W, 128)
            ae_embeddings/    <name>_ae.npy       (H, W, C_ae)   [optional]
            targets/          <name>_y.npy        (H, W)
            norm_stats.json                        GeoTessera stats
            norm_stats_ae.json                     AlphaEarth stats [optional]

    Args:
        data_dir    : root folder described above
        patch_size  : square patch side length in pixels
        split       : "train" | "val" | "test"
        split_file  : optional path to a JSON manifest with explicit splits
                      e.g. {"train": ["amazon_2020", ...], "val": [...], "test": [...]}
        use_ae      : whether to load and fuse AlphaEarth embeddings
        augment     : random rot90 + horizontal flip (training only)
    """

    def __init__(self, data_dir, patch_size=64, split="train",
                 split_file=None, use_ae=False, use_augment=False):

        self.patch_size = patch_size
        self.use_ae = use_ae ; self.augment = use_augment

        data_dir = Path(data_dir)
        emb_dir  = data_dir / "embeddings"
        ae_dir   = data_dir / "ae_embeddings"
        tgt_dir  = data_dir / "targets"

        # ── Normalization stats ───────────────────────────────────────────────
        stats_path = data_dir / "norm_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.mean = np.array(stats["mean"], dtype=np.float32)
            self.std  = np.array(stats["std"],  dtype=np.float32)
        else:
            print(f"Warning: {stats_path} not found — skipping GeoTessera normalization.")
            self.mean = None
            self.std  = None

        ae_stats_path = data_dir / "norm_stats_ae.json"
        if ae_stats_path.exists():
            with open(ae_stats_path) as f:
                ae_stats = json.load(f)
            self.ae_mean = np.array(ae_stats["mean"], dtype=np.float32)
            self.ae_std  = np.array(ae_stats["std"],  dtype=np.float32)
        else:
            if use_ae:
                print(f"Warning: {ae_stats_path} not found — skipping AE normalization.")
            self.ae_mean = None
            self.ae_std  = None

        # ── Split manifest ────────────────────────────────────────────────────
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                manifest = json.load(f)
            if split not in manifest:
                raise KeyError(f"Split '{split}' not found in {split_file}")
            names = manifest[split]
        else:
            all_names = [f.stem.replace("_x", "")
                         for f in emb_dir.glob("*_x.npy")]
            if not all_names:
                raise FileNotFoundError(f"No *_x.npy files found in {emb_dir}")
            names = self._default_split(all_names, split)

        # ── Memory-map tiles (no RAM cost until sliced) ───────────────────────
        self.tiles = []
        for name in names:
            emb_path = emb_dir / f"{name}_x.npy"
            tgt_path = tgt_dir / f"{name}_y.npy"

            if not emb_path.exists() or not tgt_path.exists():
                print(f"Warning: missing files for '{name}', skipping.")
                continue

            emb = np.load(emb_path, mmap_mode='r')   # (H, W, 128)
            tgt = np.load(tgt_path, mmap_mode='r')   # (H, W)
            ae  = None

            if use_ae:
                ae_path = ae_dir / f"{name}_ae.npy"
                if ae_path.exists():
                    ae = np.load(ae_path, mmap_mode='r')
                else:
                    print(f"Warning: AE file missing for '{name}', using GeoTessera only.")

            H, W, _ = emb.shape
            self.tiles.append({
                "name": name,
                "emb":  emb,
                "tgt":  tgt,
                "ae":   ae,
                "H":    H,
                "W":    W,
            })

        if not self.tiles:
            raise RuntimeError("No valid tiles loaded — check your data directory.")

        # ── Flat patch index: (tile_idx, row_start, col_start) ───────────────
        self.index = []
        for ti, tile in enumerate(self.tiles):
            H, W, P = tile["H"], tile["W"], patch_size
            for r in range(0, H - P, P):
                for c in range(0, W - P, P):
                    self.index.append((ti, r, c))

        print(f"BiomassDataset [{split}]: {len(self.tiles)} tiles, "
              f"{len(self.index)} patches of size {patch_size}x{patch_size}")

    # ── Magic methods ─────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ti, r, c = self.index[idx]
        P = self.patch_size
        tile = self.tiles[ti]

        x = tile["emb"][r:r+P, c:c+P, :].copy()   # (P, P, 128)
        y = tile["tgt"][r:r+P, c:c+P].copy()       # (P, P)

        # Normalize GeoTessera embedding BEFORE concatenation
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-6)

        # Optionally fuse AlphaEarth embedding
        if self.use_ae and tile["ae"] is not None:
            ae = tile["ae"][r:r+P, c:c+P, :].copy()
            if self.ae_mean is not None:
                ae = (ae - self.ae_mean) / (self.ae_std + 1e-6)
            x = np.concatenate([x, ae], axis=-1)   # (P, P, 128 + C_ae)

        # Augmentation (training only)
        if self.augment:
            x, y = self._augment(x, y)

        # HWC → CHW for PyTorch
        x = torch.from_numpy(x.copy()).permute(2, 0, 1).float()  # (C, P, P)
        y = torch.from_numpy(y.copy()).float()                    # (P, P)
        return x, y

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _augment(self, x, y): # verify x [H,W,C] ?
        """Random rot90 + horizontal flip — both applied identically to x and y."""
        k = np.random.randint(0, 4)
        x = np.rot90(x, k).copy()
        y = np.rot90(y, k).copy()
        if np.random.rand() > 0.5:
            x = np.fliplr(x).copy()
            y = np.fliplr(y).copy()
        return x, y

    @staticmethod
    def _default_split(names, split, seed=42):
        """Reproducible 70/15/15 train/val/test split."""
        rng      = np.random.default_rng(seed)
        shuffled = rng.permutation(names).tolist()
        n        = len(shuffled)
        cuts     = (int(0.70 * n), int(0.85 * n))
        splits   = {
            "train": shuffled[:cuts[0]],
            "val":   shuffled[cuts[0]:cuts[1]],
            "test":  shuffled[cuts[1]:],
        }
        if split not in splits:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        return splits[split]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(data_dir, patch_size=64, batch_size=32,
                     use_ae=False, num_workers=4, split_file=None):
    """
    Returns (train_loader, val_loader, test_loader).
    Augmentation is enabled only for the training split.
    """
    train_ds = BiomassDataset(data_dir, patch_size=patch_size, split="train",
                               split_file=split_file, use_ae=use_ae, augment=True)
    val_ds   = BiomassDataset(data_dir, patch_size=patch_size, split="val",
                               split_file=split_file, use_ae=use_ae, augment=False)
    test_ds  = BiomassDataset(data_dir, patch_size=patch_size, split="test",
                               split_file=split_file, use_ae=use_ae, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — compute stats then sanity-check the dataset
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "data"

    # Step 1: compute and save normalization stats (run once)
    print("=== Computing normalization stats ===")
    compute_normalization_stats(DATA_DIR, subdir="embeddings",    out="norm_stats.json")
    compute_normalization_stats(DATA_DIR, subdir="ae_embeddings", out="norm_stats_ae.json")

    # Step 2: sanity check
    print("\n=== Sanity check ===")
    train_loader, val_loader, test_loader = make_dataloaders(
        DATA_DIR, patch_size=64, batch_size=8, use_ae=True, num_workers=0
    )

    x_batch, y_batch = next(iter(train_loader))
    print(f"x batch: {x_batch.shape}  dtype={x_batch.dtype}")  # (8, C, 64, 64)
    print(f"y batch: {y_batch.shape}  dtype={y_batch.dtype}")  # (8, 64, 64)
    print(f"x mean={x_batch.mean():.4f}  std={x_batch.std():.4f}")  # should be ~0, ~1
    print(f"y range=[{y_batch.min():.1f}, {y_batch.max():.1f}]")    # biomass Mg/ha

# would augmentation like rotation and all make sense or just leak informations ?