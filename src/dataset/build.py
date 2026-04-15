import os
import io
import sys
import time
import random
import requests
import numpy as np
import tifffile
import ee

from urllib.error import HTTPError
from rasterio.transform import from_bounds
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
OUTPUT_DIR = "data_uniform"
AE_DIR = os.path.join(OUTPUT_DIR, "ae_embeddings")
TARGET_DIR = os.path.join(OUTPUT_DIR, "targets")

os.makedirs(AE_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

ee.Initialize(project="alexcloud-489214")

MASTER_DIM = 256
BUFFER_DEG = 0.09   # ~10km
TOTAL_SAMPLES = 10
MAX_TRIES = 10000   # safety cap

# ==============================
# UTIL FUNCTIONS
# ==============================

def sample_uniform_lat_lon():
    """Uniform sampling on sphere"""
    u = random.random()
    v = random.random()

    lat = np.degrees(np.arcsin(2 * u - 1))
    lon = 360 * v - 180
    return lat, lon

def is_on_land(lat, lon, land_geom):
    pt = ee.Geometry.Point([lon, lat])
    try:
        return land_geom.contains(pt).getInfo()
    except Exception:
        return False
    

def sample_point_in_zone(france_geom):
    pt = ee.FeatureCollection.randomPoints(
        region=france_geom,
        points=1,
        seed=random.randint(0, 1_000_000)
    ).first()

    coords = pt.geometry().coordinates().getInfo()
    lon, lat = coords
    return lat, lon







def build_geom(lat, lon):
    min_lon = lon - BUFFER_DEG
    max_lon = lon + BUFFER_DEG
    min_lat = lat - BUFFER_DEG
    max_lat = lat + BUFFER_DEG

    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat]), \
           (min_lon, min_lat, max_lon, max_lat)


def fetch_alphaearth(geom, bounds):
    ae_coll = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
               .filterBounds(geom)
               .sort('system:time_start', False))

    ae_img = ae_coll.first()
    if ae_img is None:
        return None

    ae_img = ae_img.unmask(0).toFloat()

    bands = [f"A{i:02d}" for i in range(64)]
    chunk_size = 64
    chunks = []

    for i in range(0, 64, chunk_size):
        sub = bands[i:i+chunk_size]
        chunk_img = ae_img.select(sub)

        url = chunk_img.getDownloadURL({
            'region': geom,
            'format': 'GEO_TIFF',
            'dimensions': f'{MASTER_DIM}x{MASTER_DIM}',
            'crs': 'EPSG:4326'
        })

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        with io.BytesIO(resp.content) as f:
            arr = tifffile.imread(f)
            arr = np.nan_to_num(arr, nan=0.0)

            if arr.ndim == 2:
                arr = arr[..., None]

            chunks.append(arr)

    ae = np.concatenate(chunks, axis=-1)

    # quality check
    if np.mean(np.abs(ae)) < 1e-6:
        return None

    return ae


def fetch_agb(geom, year=2020):
    agb = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
           .filterDate(f"{year}-01-01", f"{year}-12-31")
           .filterBounds(geom)
           .first())

    if agb is None:
        return None

    agb = agb.select('AGB')

    url = agb.getDownloadURL({
        'region': geom,
        'format': 'GEO_TIFF',
        'dimensions': f'{MASTER_DIM}x{MASTER_DIM}',
        'crs': 'EPSG:4326'
    })

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with io.BytesIO(resp.content) as f:
        arr = tifffile.imread(f)
        arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)

        if arr.ndim == 2:
            arr = arr[..., None]

    # quality check
    if np.mean(arr) < 1e-3:
        return None

    return arr


# ==============================
# MAIN LOOP
# ==============================
land_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") ; land_geom = land_fc.geometry()
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") ; france = countries.filter(ee.Filter.eq('country_na', 'France')).geometry()
accepted = 0 ; tries = 0

pbar = tqdm(total=TOTAL_SAMPLES, desc="Accepted samples", unit="sample")
while accepted < TOTAL_SAMPLES and tries < MAX_TRIES:
    tries += 1

    lat, lon = sample_point_in_zone(france)
    geom, bounds = build_geom(lat, lon)

    print(f"\n🌍 Try {tries} | Sampling ({lat:.4f}, {lon:.4f})")

    try:
        ae = fetch_alphaearth(geom, bounds)
        if ae is None:
            print("❌ AlphaEarth invalid")
            continue

        agb = fetch_agb(geom)
        if agb is None:
            print("❌ AGB invalid")
            continue

        # SAVE
        #name = f"sample_{accepted:05d}"
        #name = f"lat{lat:.4f}_lon{lon:.4f}"
        name = f"lat{lat:+08.4f}_lon{lon:+09.4f}_{accepted:05d}"

        np.save(os.path.join(AE_DIR, f"{name}_ae.npy"), ae)
        np.save(os.path.join(TARGET_DIR, f"{name}_y.npy"), agb)

        #print(f"✅ Accepted sample {accepted}")
        accepted += 1
        pbar.update(1)
        accept_rate = accepted / tries
        pbar.set_postfix({
            "tries": tries,
            "acc_rate": f"{accept_rate:.2f}"
        })
        print(f"✅ Accepted sample {accepted}")

        # small delay to avoid throttling
        time.sleep(1)

    except HTTPError:
        print("   ⚠️ HTTP error")
    except Exception as e:
        print(f"   ❌ Error: {e}")

pbar.close()
print("\n====================")
print(f"Done: {accepted} samples collected in {tries} tries")