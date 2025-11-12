"""
Multi-class Image Classification using LADI-v2 Classifier via Hugging Face Endpoint
Similar workflow to RescueNet YOLO but for multi-label classification instead of detection

Model identifies disaster response features from aerial images:
- bridges_any, buildings_any, buildings_affected_or_greater, buildings_minor_or_greater
- debris_any, flooding_any, flooding_structures
- roads_any, roads_damage, trees_any, trees_damage, water_any
"""

import os
import math
import json
import cv2
import time
import datetime
import numpy as np
import requests
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, firestore, storage


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

# Hugging Face Inference Endpoint
HF_ENDPOINT_URL = "https://p6def67acw3ebe9z.us-east-1.aws.endpoints.huggingface.cloud"
HF_API_KEY = os.environ.get('HF_API_KEY', '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')  # Default token provided
CONFIDENCE_THRESHOLD = 0.5  # Only save labels with confidence above this threshold

# Firestore configuration
FIRESTORE_COLLECTION = "image_classifications"
FIREBASE_CRED_PATH = "firebase-credentials.json"

# Storage bucket configuration
try:
    from config import FIREBASE_CONFIG
except Exception:
    FIREBASE_CONFIG = None

env_bucket = os.environ.get('FIREBASE_STORAGE_BUCKET')
if env_bucket:
    _raw_bucket = env_bucket
elif FIREBASE_CONFIG and isinstance(FIREBASE_CONFIG, dict):
    _raw_bucket = FIREBASE_CONFIG.get('storageBucket') or "jamaica-realtime-crisis-map.appspot.com"
else:
    _raw_bucket = "jamaica-realtime-crisis-map.appspot.com"

if isinstance(_raw_bucket, str):
    STORAGE_BUCKET = _raw_bucket.strip()
    if STORAGE_BUCKET.startswith('gs://'):
        STORAGE_BUCKET = STORAGE_BUCKET[5:]
    if STORAGE_BUCKET.startswith('https://'):
        STORAGE_BUCKET = STORAGE_BUCKET.split('/')[-1]
    STORAGE_BUCKET = STORAGE_BUCKET.strip('/')
else:
    STORAGE_BUCKET = _raw_bucket

NOAA_URL = "https://stormscdn.ngs.noaa.gov/20251031a-rgb"
TILE_FOLDER = "tiles_classification"

os.makedirs(TILE_FOLDER, exist_ok=True)


# -------------------------------------------------
# FIREBASE INITIALIZATION
# -------------------------------------------------

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        default_bucket = None
        try:
            with open(FIREBASE_CRED_PATH, 'r') as fh:
                cred_json = json.load(fh)
                project_id = cred_json.get('project_id')
                if project_id:
                    default_bucket = f"{project_id}.appspot.com"
        except Exception:
            default_bucket = None

        bucket_name_to_use = STORAGE_BUCKET or default_bucket

        if bucket_name_to_use:
            try:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name_to_use
                })
            except Exception:
                firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app(cred)

    db = firestore.client()
    try:
        bucket = storage.bucket()
        try:
            if hasattr(bucket, 'exists'):
                exists = bucket.exists()
            else:
                list(bucket.list_blobs(max_results=1))
                exists = True
        except Exception:
            exists = False

        if not exists:
            print(f"⚠ Storage bucket '{bucket.name if hasattr(bucket, 'name') else STORAGE_BUCKET}' does not exist or is inaccessible.")
            bucket = None
    except Exception as e:
        print(f"⚠ Could not access storage bucket: {e}")
        bucket = None
    print(f"✓ Firebase initialized successfully")
    if bucket is not None:
        print(f"✓ Storage bucket: {bucket.name}")
    else:
        print(f"⚠ No accessible storage bucket configured.")
except Exception as e:
    print(f"⚠ Firebase initialization failed: {e}")
    db = None
    bucket = None


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
    """Generate tiles at `target_zoom` that cover the provided bounding box."""
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
    """Download tile and return as BytesIO object."""
    url = f"{NOAA_URL}/{z}/{x}/{y}"
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            image_data = io.BytesIO(r.content)
            return image_data
        else:
            log(f"Missing tile: {url} ({r.status_code})")
    except requests.RequestException as e:
        log(f"Error downloading {url}: {e}")
    return None


# -------------------------------------------------
# HUGGING FACE ENDPOINT INFERENCE
# -------------------------------------------------

def classify_image_hf(image_bytes):
    """
    Send image to Hugging Face endpoint for classification.
    For image tasks, send the data as binary with the appropriate mime type.
    
    Args:
        image_bytes: bytes of the image
    Returns:
        List of dicts with 'label' and 'score' keys, or None on error
    """
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "image/png"  # Send as binary image
    }
    
    try:
        response = requests.post(HF_ENDPOINT_URL, headers=headers, data=image_bytes, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            log(f"HF API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        log(f"Error calling HF endpoint: {e}")
        return None


# -------------------------------------------------
# STORAGE UPLOAD
# -------------------------------------------------

def upload_image_to_storage(image_data, z, x, y):
    """Upload original tile to Firebase Storage."""
    if bucket is None:
        return None
    
    try:
        filename = f"classification_originals/z{z}/{x}_{y}.png"
        blob = bucket.blob(filename)
        image_data.seek(0)
        blob.upload_from_file(image_data, content_type='image/png')
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return f"gs://{bucket.name}/{filename}"
    except Exception as e:
        log(f"Upload failed for {z}/{x}/{y}: {e}")
        return None


# -------------------------------------------------
# FIRESTORE SAVE
# -------------------------------------------------

def save_classification_to_firestore(classification_data):
    """Save classification result to Firestore."""
    if db is None:
        return False
    
    try:
        classification_data['timestamp'] = firestore.SERVER_TIMESTAMP
        classification_data['created_at'] = datetime.datetime.now().isoformat()
        
        db.collection(FIRESTORE_COLLECTION).add(classification_data)
        return True
    except Exception as e:
        log(f"Error saving to Firestore: {e}")
        return False


# -------------------------------------------------
# INFERENCE PIPELINE
# -------------------------------------------------

def run_inference(tile_data, z, x, y, overlap=0):
    """Run classification on a single tile."""
    if tile_data is None:
        return []
    
    # Upload original tile
    original_image_url = None
    try:
        if bucket is not None:
            tile_data.seek(0)
            bytes_data = tile_data.read()
            if bytes_data:
                original_image_url = upload_image_to_storage(io.BytesIO(bytes_data), z, x, y)
                tile_data = io.BytesIO(bytes_data)  # Recreate for next use
    except Exception as e:
        log(f"Original upload failed for {z}/{x}/{y}: {e}")
    
    # Get tile bounds
    lat_n, lon_w, lat_s, lon_e = tile_to_bounds(x, y, z)
    center_lat = (lat_n + lat_s) / 2.0
    center_lon = (lon_w + lon_e) / 2.0
    
    # Run classification
    try:
        tile_data.seek(0)
        image_bytes = tile_data.read()
        
        results = classify_image_hf(image_bytes)
        
        if results is None:
            log(f"No classification results for {z}/{x}/{y}")
            return []
        
        # Results format: [{'label': 'buildings_any', 'score': 0.9994}, ...]
        # Filter results by confidence threshold
        significant_results = [(item['label'], item['score']) for item in results if item['score'] >= CONFIDENCE_THRESHOLD]
        
        if not significant_results:
            log(f"No significant labels (>= {CONFIDENCE_THRESHOLD}) for {z}/{x}/{y}")
            return []
        
        # Get top prediction
        top_result = max(results, key=lambda x: x['score'])
        top_label = top_result['label']
        top_score = top_result['score']
        
        # Create feature with all predictions
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [center_lon, center_lat]
            },
            "properties": {
                "primary_label": top_label,
                "primary_confidence": float(top_score),
                "tile": f"{z}/{x}/{y}",
                "zoom_level": z,
                "tile_x": x,
                "tile_y": y,
                "all_predictions": {item['label']: float(item['score']) for item in results},
                "significant_labels": {label: float(score) for label, score in significant_results},
                "original_image_url": original_image_url
            }
        }
        
        # Save to Firestore
        if save_classification_to_firestore(feature):
            log(f"Classification: {top_label} ({top_score:.2%}) + {len(significant_results)-1} other labels @ {z}/{x}/{y}")
            return [feature]
        
        return []
        
    except Exception as e:
        log(f"Error during inference on {z}/{x}/{y}: {e}")
        return []


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

def process_zoom20_from_zoom15(lat, lon, force=False, overlap=0):
    """Process tiles in a zoom-15 area."""
    overall_start = time.time()
    log(f"Starting classification pipeline at lat={lat}, lon={lon}")
    log(f"Firestore collection: {FIRESTORE_COLLECTION}")

    bbox = zoom15_bbox(lat, lon)
    log(f"Zoom-15 bounding box:")
    log(f"  Top-Left: lat={bbox['north']:.6f}, lon={bbox['west']:.6f}")
    log(f"  Bottom-Right: lat={bbox['south']:.6f}, lon={bbox['east']:.6f}")

    tiles = generate_zoom20_tiles(bbox["north"], bbox["west"], bbox["south"], bbox["east"], target_zoom=20, overlap=overlap)
    total_classifications = 0

    for z, x, y in tiles:
        # Per-tile duplicate check
        tile_key = f"{z}/{x}/{y}"
        already_processed = False
        try:
            if db is not None:
                docs = list(db.collection(FIRESTORE_COLLECTION).where('properties.tile', '==', tile_key).limit(1).stream())
                if docs:
                    if not force:
                        log(f"Skipping tile {tile_key} - already processed")
                        already_processed = True
                    else:
                        log(f"Force reprocessing tile {tile_key}")
        except Exception as e:
            log(f"Tile check failed for {tile_key}: {e} -- will process anyway")

        if already_processed:
            continue

        tile_data = download_noaa_tile(z, x, y)
        if not tile_data:
            continue
        
        results = run_inference(tile_data, z, x, y, overlap=overlap)
        total_classifications += len(results)

    log(f"Total classifications: {total_classifications}")
    log(f"Total runtime: {time.time()-overall_start:.2f}s")
    
    if db:
        log(f"✓ All classifications saved to Firestore collection: {FIRESTORE_COLLECTION}")
    else:
        log(f"⚠ Firestore not available - no classifications saved")
    
    return total_classifications


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    total_count = process_zoom20_from_zoom15(lat=18.0384131, lon=-77.8600417)
    log(f"Pipeline completed. Total classifications: {total_count}")
