"""
NOAA tile pipeline (Zoom 15 -> Zoom 20) with Flood Segmentation:
- Flood Segmentation: identifies flooded areas from satellite imagery
"""

import os
import math
import json
import cv2
import io
import base64
import time
import datetime
import firebase_admin
from firebase_admin import credentials, storage as fb_storage, firestore as fb_firestore
import numpy as np
import requests
from roboflow import Roboflow
import supervision as sv
import numpy as np


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

ROBOFLOW_API_KEY = "nLepKB2uhcrGMbdJJPdl"

# Model configuration
WORKSPACE_NAME = None  # Will use default workspace
PROJECT_NAME = "flood-1sljl"
MODEL_VERSION = 2

NOAA_URL = "https://stormscdn.ngs.noaa.gov/20251031a-rgb"
TILE_FOLDER = "tiles_flood"
OUTPUT_DIR = "outputs_flood"

os.makedirs(TILE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------
# TIMING HELPERS
# -------------------------------------------------

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{timestamp()}] {msg}")


# -------------------------------------------------
# TILE MATH
# -------------------------------------------------

def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def tile_to_bounds(x, y, z):
    n = 2.0 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lat_n, lon_w, lat_s, lon_e

def zoom15_bbox(lat, lon):
    x15, y15 = latlon_to_tile(lat, lon, 15)
    lat_n, lon_w, lat_s, lon_e = tile_to_bounds(x15, y15, 15)
    return {"north": lat_n, "west": lon_w, "south": lat_s, "east": lon_e}

def generate_zoom20_tiles(lat_top, lon_left, lat_bottom, lon_right, target_zoom=20, overlap=0):
    """
    Generate tiles at `target_zoom` that cover the provided bounding box.

    overlap: fraction (e.g. 0.25) by which to expand the input bbox (both dims) before
    computing tiles. Expanding prevents features near tile edges from being cut off.
    """
    # Expand bbox by overlap fraction (split evenly on each side)
    lat_span = abs(lat_top - lat_bottom)
    lon_span = abs(lon_right - lon_left)
    pad_lat = lat_span * overlap
    pad_lon = lon_span * overlap

    lat_top_exp = lat_top + (pad_lat / 2.0)
    lat_bottom_exp = lat_bottom - (pad_lat / 2.0)
    lon_left_exp = lon_left - (pad_lon / 2.0)
    lon_right_exp = lon_right + (pad_lon / 2.0)

    z = target_zoom
    x_min, y_min = latlon_to_tile(lat_top_exp, lon_left_exp, z)
    x_max, y_max = latlon_to_tile(lat_bottom_exp, lon_right_exp, z)
    x_start, x_end = sorted([x_min, x_max])
    y_start, y_end = sorted([y_min, y_max])
    tiles = [(z, x, y) for x in range(x_start, x_end + 1) for y in range(y_start, y_end + 1)]
    log(f"Zoom-{z} tile range with overlap={overlap}: x={x_start}-{x_end}, y={y_start}-{y_end}, total={len(tiles)}")
    return tiles


# -------------------------------------------------
# NOAA TILE DOWNLOAD
# -------------------------------------------------

def download_noaa_tile(z, x, y):
    url = f"{NOAA_URL}/{z}/{x}/{y}"
    path = f"{TILE_FOLDER}/melissa_z{z}_{x}_{y}.png"
    if os.path.exists(path):
        return path
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        else:
            log(f"Missing tile: {url} ({r.status_code})")
    except requests.RequestException as e:
        log(f"Error downloading {url}: {e}")
    return None


def compose_padded_tile_from_paths(z, x, y, overlap=0):
    """Compose a padded tile from neighboring saved tile files (paths returned by download_noaa_tile).
    Returns (canvas_bgr, (x_start,x_end,y_start,y_end), tile_w, tile_h) or (None, None, None, None)
    """
    # get center path
    center_path = download_noaa_tile(z, x, y)
    if not center_path or not os.path.exists(center_path):
        return None, None, None, None

    center_img = cv2.imread(center_path)
    if center_img is None:
        return None, None, None, None

    tile_h, tile_w = center_img.shape[:2]
    pad_x = int(tile_w * overlap)
    pad_y = int(tile_h * overlap)

    canvas_w = tile_w + pad_x * 2
    canvas_h = tile_h + pad_y * 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx = x + dx
            ny = y + dy
            p = download_noaa_tile(z, nx, ny)
            if not p or not os.path.exists(p):
                continue
            img = cv2.imread(p)
            if img is None:
                continue

            off_x = pad_x + dx * tile_w
            off_y = pad_y + dy * tile_h
            x0 = max(0, off_x)
            y0 = max(0, off_y)
            x1 = min(canvas_w, off_x + tile_w)
            y1 = min(canvas_h, off_y + tile_h)

            src_x0 = max(0, -off_x)
            src_y0 = max(0, -off_y)
            src_x1 = src_x0 + (x1 - x0)
            src_y1 = src_y0 + (y1 - y0)

            canvas[y0:y1, x0:x1] = img[src_y0:src_y1, src_x0:src_x1]

    x_start = pad_x
    x_end = pad_x + tile_w
    y_start = pad_y
    y_end = pad_y + tile_h

    return canvas, (x_start, x_end, y_start, y_end), tile_w, tile_h


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

log("Initializing Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(PROJECT_NAME)
flood_model = project.version(MODEL_VERSION).model

log(f"Flood Segmentation model loaded successfully")
mask_annotator = sv.MaskAnnotator(opacity=0.6, color=(93,151,231))


# -----------------------------
# Firebase (Storage + Firestore)
# -----------------------------
firebase_app = None
firestore_client = None
storage_bucket = None


def _normalize_bucket_name(name: str):
    if not name:
        return None
    name = name.strip()
    # strip gs:// prefix and any trailing slashes or https://storage.googleapis.com/
    if name.startswith("gs://"):
        name = name[5:]
    if name.startswith("https://"):
        # try to extract bucket from storage URL
        parts = name.split('/')
        if len(parts) >= 3:
            name = parts[2]
    name = name.rstrip('/')
    return name


def init_firebase():
    global firebase_app, firestore_client, storage_bucket
    if firebase_app is not None:
        return
    try:
        # prefer explicit service account via env var
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            # If user provided FIREBASE_STORAGE_BUCKET env var, pass into options
            bucket_name = os.getenv('FIREBASE_STORAGE_BUCKET') or None
            opts = {'storageBucket': _normalize_bucket_name(bucket_name)} if bucket_name else None
            if not firebase_admin._apps:
                firebase_app = firebase_admin.initialize_app(cred, opts)
        else:
            # try to initialize application default
            if not firebase_admin._apps:
                firebase_app = firebase_admin.initialize_app()

        firestore_client = fb_firestore.client()

        # determine bucket
        cfg_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
        # fallback to config.FIREBASE_CONFIG if present
        try:
            import config
            if not cfg_bucket and hasattr(config, 'FIREBASE_CONFIG'):
                cfg_bucket = config.FIREBASE_CONFIG.get('storageBucket')
        except Exception:
            pass

        if cfg_bucket:
            cfg_bucket = _normalize_bucket_name(cfg_bucket)
        # attempt to set storage bucket
        try:
            storage_bucket = fb_storage.bucket(cfg_bucket) if cfg_bucket else fb_storage.bucket()
            # try existence check
            try:
                exists = getattr(storage_bucket, 'exists', None)
                if callable(exists):
                    if not storage_bucket.exists():
                        log(f"Storage bucket '{cfg_bucket}' does not appear to exist.")
                        storage_bucket = None
            except Exception:
                # skip precise existence check if method not available
                pass
        except Exception as e:
            log(f"Unable to initialize storage bucket: {e}")
            storage_bucket = None

        log("Firebase initialized (Firestore + Storage)" if firestore_client and storage_bucket else "Firebase initialized (partial)" )
    except Exception as e:
        log(f"Firebase init error: {e}")
        firebase_app = None
        firestore_client = None
        storage_bucket = None


def upload_bytes_to_storage(data_bytes: bytes, dest_path: str, content_type: str = 'image/jpeg'):
    """Upload in-memory bytes to Firebase Storage and make public. Returns public URL or None."""
    if storage_bucket is None:
        return None
    try:
        blob = storage_bucket.blob(dest_path)
        blob.upload_from_string(data_bytes, content_type=content_type)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            # fallback: return gs:// path
            return f"gs://{storage_bucket.name}/{dest_path}"
    except Exception as e:
        log(f"Upload failed for {dest_path}: {e}")
        return None


def upload_cv2_image(image, dest_path, fmt='jpg', quality=85):
    """Encode a cv2 image and upload to storage; returns public URL or None."""
    ext = '.jpg' if fmt == 'jpg' else '.png'
    # Ensure image is a numpy array in BGR format with 3 channels for JPEG
    arr = image
    try:
        import numpy as _np
        # If PIL Image passed in, convert to numpy BGR
        try:
            from PIL import Image as _PILImage
        except Exception:
            _PILImage = None

        if _PILImage is not None and isinstance(image, _PILImage.Image):
            # Convert RGBA -> RGB if needed
            if image.mode in ('RGBA', 'LA'):
                image = image.convert('RGB')
            rgb = _np.array(image)
            # Convert RGB to BGR for cv2
            arr = rgb[:, :, ::-1]
        else:
            # assume numpy array
            if isinstance(image, _np.ndarray):
                arr = image
                # grayscale -> BGR
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                # BGRA -> BGR
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    except Exception:
        # fallback: use image as-is
        arr = image

    if fmt == 'jpg':
        ok, buf = cv2.imencode(ext, arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        content_type = 'image/jpeg'
    else:
        ok, buf = cv2.imencode(ext, arr)
        content_type = 'image/png'
    if not ok:
        return None
    return upload_bytes_to_storage(buf.tobytes(), dest_path, content_type=content_type)



# -------------------------------------------------
# FLOOD SEGMENTATION INFERENCE
# -------------------------------------------------

def run_inference(tile_path, z, x, y, overlap=0):
    # Process single tile image (no stitching/padding)
    try:
        image = cv2.imread(tile_path)
        if image is None:
            log(f"Could not read tile image at {tile_path}")
            return []

        tile_h, tile_w = image.shape[0], image.shape[1]
        x_start, x_end = 0, tile_w
        y_start, y_end = 0, tile_h

        lat_n, lon_w, lat_s, lon_e = tile_to_bounds(x, y, z)

        # Run model on single tile image
        prediction = flood_model.predict(image)
        result_json = prediction.json()

        # Build flood mask on the tile
        flood_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        if isinstance(result_json, dict) and 'predictions' in result_json:
            for pred in result_json['predictions']:
                if 'points' in pred:
                    points = pred['points']
                    pts = np.array([[int(p['x']), int(p['y'])] for p in points], np.int32)
                    cv2.fillPoly(flood_mask, [pts], 1)

        # Compute flood ratio over the tile
        flood_ratio = float(np.count_nonzero(flood_mask) / flood_mask.size) if flood_mask.size > 0 else 0.0

        if flood_ratio <= 0.0:
            log(f"No flooding detected ({flood_ratio:.2%}) @ {z}/{x}/{y}; skipping upload and Firestore write.")
            return []

        # Create annotated image for the tile
        annotated = image.copy()
        mask_cropped = flood_mask.copy()
        flood_overlay = np.zeros_like(annotated)
        flood_overlay[mask_cropped > 0] = [255, 0, 0]
        annotated = cv2.addWeighted(annotated, 0.7, flood_overlay, 0.3, 0)
        text = f"Flood Coverage: {flood_ratio:.1%}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        try:
            init_firebase()
        except Exception:
            pass

        image_url = None
        doc_ref = None
        orig_url = None

        try:
            if storage_bucket is not None:
                # upload original tile if available
                try:
                    if os.path.exists(tile_path):
                        with open(tile_path, 'rb') as f:
                            original_bytes = f.read()
                        orig_blob = f"flood_originals/z{z}/{x}_{y}.png"
                        orig_url = upload_bytes_to_storage(original_bytes, orig_blob, content_type='image/png')
                    else:
                        orig_url = None
                except Exception as e:
                    log(f"Original upload failed for {z}/{x}/{y}: {e}")
                    orig_url = None

                annotated_blob_path = f"flood_annotations/z{z}/{x}_{y}.jpg"
                image_url = upload_cv2_image(annotated, annotated_blob_path, fmt='jpg', quality=85)

        except Exception as e:
            log(f"Upload exception: {e}")

        preds_count = len(result_json.get('predictions', [])) if isinstance(result_json, dict) else 0
        log(f"Flood: {flood_ratio:.2%} @ {z}/{x}/{y}")
        log(f"  Predictions: {preds_count}")

        features = [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon_w, lat_n],
                    [lon_e, lat_n],
                    [lon_e, lat_s],
                    [lon_w, lat_s],
                    [lon_w, lat_n]
                ]]
            },
            "properties": {
                "label": "flood_extent",
                "flood_ratio": flood_ratio,
                "tile": f"{z}/{x}/{y}",
                "image_url": image_url,
                "original_image_url": orig_url,
                "zoom_level": z,
                "timestamp": timestamp()
            }
        }]

        try:
            if firestore_client is not None:
                coll = firestore_client.collection('flood_extents')
                def to_native(obj):
                    if isinstance(obj, np.generic):
                        return obj.item()
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, dict):
                        return {str(k): to_native(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [to_native(x) for x in obj]
                    return obj

                native_doc = to_native({ 'geometry': features[0]['geometry'], 'properties': features[0]['properties'] })
                doc_ref = coll.add(native_doc)
                log(f"Wrote flood document to Firestore (tile={z}/{x}/{y})")
        except Exception as e:
            log(f"Firestore write failed: {e}")

        return features
    except Exception as e:
        log(f"ERROR during inference on {z}/{x}/{y}: {e}")
        return []
        
    except Exception as e:
        log(f"ERROR during inference on {z}/{x}/{y}: {e}")
        return []


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

def process_zoom20_from_zoom15(lat, lon, force=False, overlap=0):
    overall_start = time.time()
    log(f"Starting flood segmentation pipeline at lat={lat}, lon={lon}")

    bbox = zoom15_bbox(lat, lon)
    log(f"Zoom-15 bounding box:")
    log(f"  Top-Left: lat={bbox['north']:.6f}, lon={bbox['west']:.6f}")
    log(f"  Bottom-Right: lat={bbox['south']:.6f}, lon={bbox['east']:.6f}")

    tiles = generate_zoom20_tiles(bbox["north"], bbox["west"], bbox["south"], bbox["east"], target_zoom=20, overlap=overlap)
    all_features = []

    for z, x, y in tiles:
        path = download_noaa_tile(z, x, y)
        if not path:
            continue
        feats = run_inference(path, z, x, y, overlap=overlap)
        all_features.extend(feats)

    # We do not save local backups. The per-tile run_inference() function
    # uploads images and writes metadata to Firestore when configured.
    log(f"Total tiles processed: {len(all_features)}")
    log(f"Total runtime: {time.time()-overall_start:.2f}s")
    return all_features


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    GEOJSON_PATH = process_zoom20_from_zoom15(lat=18.0384131, lon=-77.8600417)
    log(f"Pipeline completed. Output GeoJSON: {GEOJSON_PATH}")
