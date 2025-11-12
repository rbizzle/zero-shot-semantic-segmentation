"""
NOAA tile pipeline (Zoom 15 -> Zoom 20) with RescueNet YOLO:
- RescueNet YOLO: detects objects/infrastructure/damage
- Outputs detections directly to Firestore 'building_damage_pins' collection
"""

import os
import math
import json
import cv2
import time
import datetime
import numpy as np
import requests
from inference import get_model
import supervision as sv
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import io


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

ROBOFLOW_API_KEY = "nLepKB2uhcrGMbdJJPdl"

# Models
YOLO_MODEL_ID = "rescuenet-yolonas-ignc2/1"

# Firestore configuration
FIRESTORE_COLLECTION = "building_damage_pins"
FIREBASE_CRED_PATH = "firebase-credentials.json"  # Update with your Firebase credentials file
# Allow overriding the storage bucket via environment variable for flexibility and safety
# Read and normalize FIREBASE_STORAGE_BUCKET env var (accept gs:// or full urls)
try:
    from config import FIREBASE_CONFIG
except Exception:
    FIREBASE_CONFIG = None

# Priority: environment variable > config.py FIREBASE_CONFIG['storageBucket'] > hardcoded default
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
TILE_FOLDER = "tiles"

os.makedirs(TILE_FOLDER, exist_ok=True)


# -------------------------------------------------
# FIREBASE INITIALIZATION
# -------------------------------------------------

# Initialize Firebase Admin SDK
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        # Try to determine a reasonable default bucket if none or wrong
        default_bucket = None
        try:
            # Attempt to read project_id from the credentials file
            with open(FIREBASE_CRED_PATH, 'r') as fh:
                cred_json = json.load(fh)
                project_id = cred_json.get('project_id')
                if project_id:
                    default_bucket = f"{project_id}.appspot.com"
        except Exception:
            default_bucket = None

        # Prefer explicitly configured STORAGE_BUCKET, otherwise fall back to project_id derived bucket
        bucket_name_to_use = STORAGE_BUCKET or default_bucket

        # Initialize app with the chosen storage bucket (may still fail if bucket doesn't exist)
        if bucket_name_to_use:
            try:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name_to_use
                })
            except Exception:
                # Try initializing without storageBucket to at least get Firestore access
                firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app(cred)

    db = firestore.client()
    try:
        bucket = storage.bucket()
        # Verify the bucket actually exists on the server to avoid 404 on upload
        try:
            if hasattr(bucket, 'exists'):
                exists = bucket.exists()
            else:
                # Fallback: try listing blobs (will raise if bucket missing)
                list(bucket.list_blobs(max_results=1))
                exists = True
        except Exception:
            exists = False

        if not exists:
            print(f"⚠ Storage bucket '{bucket.name if hasattr(bucket, 'name') else STORAGE_BUCKET}' does not exist or is inaccessible.")
            bucket = None
    except Exception as e:
        # If the configured/default bucket cannot be initialized, set bucket to None and warn
        print(f"⚠ Could not access storage bucket: {e}")
        bucket = None
    print(f"✓ Firebase initialized successfully")
    if bucket is not None:
        print(f"✓ Storage bucket: {bucket.name}")
    else:
        print(f"⚠ No accessible storage bucket configured. Set STORAGE_BUCKET to a valid bucket name.")
except Exception as e:
    print(f"⚠ Firebase initialization failed: {e}")
    print(f"  Will save to local GeoJSON only")
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
    
    # Download tile without saving to disk - process in memory
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            # Return image data directly instead of saving to file
            import io
            from PIL import Image
            image_data = io.BytesIO(r.content)
            return image_data
        else:
            log(f"Missing tile: {url} ({r.status_code})")
    except requests.RequestException as e:
        log(f"Error downloading {url}: {e}")
    return None


def compose_padded_tile(z, x, y, overlap=0):
    """Download the central tile and neighboring tiles and compose a padded image.
    Returns (padded_image(cv2 BGR), (x_start, x_end, y_start, y_end), tile_w, tile_h)
    where the start/end define the central (unpadded) tile region inside the padded image.
    """
    # helper to get bytesio for a tile
    def _get_bytes(z_, x_, y_):
        try:
            return download_noaa_tile(z_, x_, y_)
        except Exception:
            return None

    center_bytes = _get_bytes(z, x, y)
    if center_bytes is None:
        return None, None, None, None

    from PIL import Image
    import numpy as _np

    # load center to determine tile size
    center_bytes.seek(0)
    pil_center = Image.open(center_bytes).convert('RGB')
    center_arr = _np.array(pil_center)
    tile_h, tile_w = center_arr.shape[0], center_arr.shape[1]

    pad_x = int(tile_w * overlap)
    pad_y = int(tile_h * overlap)

    canvas_w = tile_w + pad_x * 2
    canvas_h = tile_h + pad_y * 2

    # initialize canvas (RGB -> convert to BGR later)
    canvas = _np.zeros((canvas_h, canvas_w, 3), dtype=_np.uint8)

    # place tiles in grid around center
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx = x + dx
            ny = y + dy
            bytes_obj = _get_bytes(z, nx, ny)
            if bytes_obj is None:
                continue
            try:
                bytes_obj.seek(0)
                pil = Image.open(bytes_obj).convert('RGB')
                arr = _np.array(pil)
            except Exception:
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

            canvas[y0:y1, x0:x1] = arr[src_y0:src_y1, src_x0:src_x1]

    # convert RGB->BGR for cv2 compatibility
    canvas_bgr = canvas[:, :, ::-1]

    x_start = pad_x
    x_end = pad_x + tile_w
    y_start = pad_y
    y_end = pad_y + tile_h

    return canvas_bgr, (x_start, x_end, y_start, y_end), tile_w, tile_h


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

log("Loading RescueNet YOLO model...")
yolo_load_start = time.time()
yolo_model = get_model(model_id=YOLO_MODEL_ID, api_key=ROBOFLOW_API_KEY)
log(f"YOLO model loaded in {time.time()-yolo_load_start:.2f}s")


# -------------------------------------------------
# THUMBNAIL CREATION AND UPLOAD
# -------------------------------------------------

def create_and_upload_thumbnail(image, bbox, label, z, x, y, detection_index):
    """
    Crop detection from image, compress as thumbnail, upload to Firebase Storage
    Returns the public URL of the uploaded image
    """
    if bucket is None:
        return None
    
    try:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding around detection (10% on each side)
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)
        
        # Crop the detection
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if cropped.size == 0:
            return None
        
        # Convert to PIL Image
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        
        # Resize to thumbnail size (max 200px on longest side)
        max_size = 200
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Compress to JPEG in memory
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        img_byte_arr.seek(0)
        
        # Generate unique filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"detections/{z}/{x}/{y}/{label}_{detection_index}_{timestamp}.jpg"
        
        # Upload to Firebase Storage
        blob = bucket.blob(filename)
        blob.upload_from_file(img_byte_arr, content_type='image/jpeg')
        
        # Make publicly accessible
        blob.make_public()
        
        # Return public URL
        return blob.public_url
        
    except Exception as e:
        log(f"Error creating thumbnail: {e}")
        return None


# -------------------------------------------------
# FIRESTORE HELPER
# -------------------------------------------------

def save_detection_to_firestore(detection_data):
    """Save a single detection to Firestore"""
    if db is None:
        return False
    
    try:
        # Add timestamp
        detection_data['timestamp'] = firestore.SERVER_TIMESTAMP
        detection_data['created_at'] = datetime.datetime.now().isoformat()
        
        # Add to Firestore collection
        doc_ref = db.collection(FIRESTORE_COLLECTION).add(detection_data)
        return True
    except Exception as e:
        log(f"Error saving to Firestore: {e}")
        return False


# -------------------------------------------------
# YOLO INFERENCE
# -------------------------------------------------

def run_inference(tile_data, z, x, y, overlap=0):
    # Load image from BytesIO object
    from PIL import Image
    import io
    if tile_data is None:
        return []

    # Prepare original image upload (upload original tile once per tile)
    original_image_url = None
    try:
        if bucket is not None:
            try:
                tile_data.seek(0)
            except Exception:
                pass
            try:
                bytes_data = tile_data.read() if hasattr(tile_data, 'read') else None
                if bytes_data is None:
                    # fallback: try to get value attribute
                    bytes_data = getattr(tile_data, 'getvalue', lambda: None)()
                if bytes_data:
                    filename = f"originals/z{z}/{x}_{y}.png"
                    blob = bucket.blob(filename)
                    # upload raw bytes
                    blob.upload_from_string(bytes_data, content_type='image/png')
                    try:
                        blob.make_public()
                        original_image_url = blob.public_url
                    except Exception:
                        original_image_url = f"gs://{bucket.name}/{filename}"
            except Exception as e:
                log(f"Original upload failed for {z}/{x}/{y}: {e}")
    except Exception:
        original_image_url = None

    # Load single tile image (no stitching/padding)
    try:
        try:
            tile_data.seek(0)
        except Exception:
            pass
        pil_image = Image.open(tile_data).convert('RGB')
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        tile_h, tile_w = image.shape[0], image.shape[1]
        x_start, x_end = 0, tile_w
        y_start, y_end = 0, tile_h
        lat_n, lon_w, lat_s, lon_e = tile_to_bounds(x, y, z)

        # --- YOLO object detection on single tile ---
        results = yolo_model.infer(image)[0]
        detections = sv.Detections.from_inference(results)
    except Exception as e:
        log(f"Tile inference failed for {z}/{x}/{y}: {e}")
        return []

    saved_count = 0
    
    for idx, (xyxy, label, conf) in enumerate(zip(detections.xyxy, detections.data["class_name"], detections.confidence)):
        # Map center to lon/lat using coordinates relative to the tile
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        cx_rel = cx - x_start
        cy_rel = cy - y_start
        lon = lon_w + (lon_e - lon_w) * (cx_rel / float(tile_w))
        lat = lat_n + (lat_s - lat_n) * (cy_rel / float(tile_h))

        # Convert bbox to tile coordinate space
        x1 = max(0.0, float(xyxy[0] - x_start))
        y1 = max(0.0, float(xyxy[1] - y_start))
        x2 = min(float(tile_w), float(xyxy[2] - x_start))
        y2 = min(float(tile_h), float(xyxy[3] - y_start))

        # Create and upload thumbnail using the original tile image (crop from central region)
        try:
            # extract original tile image from padded image
            orig_img = image[int(y_start):int(y_end), int(x_start):int(x_end)].copy()
            image_url = create_and_upload_thumbnail(orig_img, (x1, y1, x2, y2), label, z, x, y, idx)
        except Exception:
            image_url = None

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "label": label,
                "confidence": float(conf),
                "tile": f"{z}/{x}/{y}",
                "zoom_level": z,
                "tile_x": x,
                "tile_y": y,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            }
        }

        # Add image URL if available
        if image_url:
            feature["properties"]["image_url"] = image_url

        # Add original tile reference if available
        if original_image_url:
            feature["properties"]["original_image_url"] = original_image_url

        # Save to Firestore
        if save_detection_to_firestore(feature):
            saved_count += 1

    log(f"{len(detections)} detections @ {z}/{x}/{y} | {saved_count} saved to Firestore")
    return len(detections)


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

def process_zoom20_from_zoom15(lat, lon, force=False, overlap=0):
    overall_start = time.time()
    log(f"Starting pipeline at lat={lat}, lon={lon}")
    log(f"Firestore collection: {FIRESTORE_COLLECTION}")

    bbox = zoom15_bbox(lat, lon)
    log(f"Zoom-15 bounding box:")
    log(f"  Top-Left: lat={bbox['north']:.6f}, lon={bbox['west']:.6f}")
    log(f"  Bottom-Right: lat={bbox['south']:.6f}, lon={bbox['east']:.6f}")

    tiles = generate_zoom20_tiles(bbox["north"], bbox["west"], bbox["south"], bbox["east"], target_zoom=20, overlap=overlap)
    total_detections = 0

    for z, x, y in tiles:
        # Per-tile duplicate check: skip if this tile was already processed and stored in Firestore
        tile_key = f"{z}/{x}/{y}"
        already_processed = False
        try:
            if db is not None:
                # look for any document with properties.tile equal to this tile
                docs = list(db.collection(FIRESTORE_COLLECTION).where('properties.tile', '==', tile_key).limit(1).stream())
                if docs:
                    if not force:
                        log(f"Skipping tile {tile_key} - already processed (found existing Firestore document)")
                        already_processed = True
                    else:
                        log(f"Force reprocessing tile {tile_key} despite existing Firestore document")
        except Exception as e:
            # Firestore may be unavailable or query failed; proceed with processing
            log(f"Tile check failed for {tile_key}: {e} -- will process anyway")

        if already_processed:
            continue

        tile_data = download_noaa_tile(z, x, y)
        if not tile_data:
            continue
        detection_count = run_inference(tile_data, z, x, y, overlap=overlap)
        total_detections += detection_count

    log(f"Total detections: {total_detections}")
    log(f"Total runtime: {time.time()-overall_start:.2f}s")
    
    if db:
        log(f"✓ All detections saved to Firestore collection: {FIRESTORE_COLLECTION}")
    else:
        log(f"⚠ Firestore not available - no detections saved")
    
    return total_detections


# -------------------------------------------------
# BACKFILL HELPERS
# -------------------------------------------------
def backfill_detection(doc_id, doc_dict, overwrite=False):
    """Re-generate and upload a thumbnail for a single Firestore detection document.

    Args:
        doc_id: Firestore document id
        doc_dict: The document dictionary (as returned by to_dict())
        overwrite: If True, re-upload even if image_url exists

    Returns:
        dict with status and image_url or error
    """
    if db is None:
        return {'ok': False, 'error': 'Firestore not initialized'}

    props = doc_dict.get('properties', {})
    if not overwrite and props.get('image_url'):
        return {'ok': True, 'skipped': True, 'reason': 'already_has_image_url'}

    # Need bbox and tile info
    bbox = props.get('bbox')
    tile = props.get('tile')  # expected like '19/2345/3456'
    label = props.get('label', 'unknown')

    if not bbox or not tile:
        return {'ok': False, 'error': 'missing bbox or tile in document properties'}

    try:
        z_str, x_str, y_str = tile.split('/')
        z = int(z_str)
        x = int(x_str)
        y = int(y_str)
    except Exception as e:
        return {'ok': False, 'error': f'invalid tile format: {e}'}

    # Download tile
    tile_data = download_noaa_tile(z, x, y)
    if not tile_data:
        return {'ok': False, 'error': 'could_not_download_tile'}

    # Load image
    try:
        from PIL import Image
        import io
        pil_image = Image.open(tile_data)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {'ok': False, 'error': f'failed_to_load_tile_image: {e}'}

    # Build bbox tuple
    try:
        x1 = int(props['bbox']['x1'])
        y1 = int(props['bbox']['y1'])
        x2 = int(props['bbox']['x2'])
        y2 = int(props['bbox']['y2'])
        bbox_tuple = (x1, y1, x2, y2)
    except Exception as e:
        return {'ok': False, 'error': f'invalid bbox values: {e}'}

    # Create and upload thumbnail
    try:
        image_url = create_and_upload_thumbnail(image, bbox_tuple, label, z, x, y, detection_index=0)
        if not image_url:
            return {'ok': False, 'error': 'thumbnail_upload_failed_or_bucket_missing'}

        # Update Firestore document with new image_url
        try:
            db.collection(FIRESTORE_COLLECTION).document(doc_id).update({'properties.image_url': image_url})
        except Exception as e:
            return {'ok': False, 'error': f'uploaded_but_failed_to_update_firestore: {e}', 'image_url': image_url}

        return {'ok': True, 'image_url': image_url}
    except Exception as e:
        return {'ok': False, 'error': f'thumbnail_creation_error: {e}'}


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    total_count = process_zoom20_from_zoom15(lat=18.0384131, lon=-77.8600417)
    log(f"Pipeline completed. Total detections: {total_count}")
