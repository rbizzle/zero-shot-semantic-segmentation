"""
Flask web app with GPT-like chat interface and OpenStreetMap view
Displays building damage detections from Firestore on the map
"""

from flask import Flask, render_template, request, jsonify, session
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import json
import uuid
try:
    from config import FIREBASE_CONFIG
except Exception:
    FIREBASE_CONFIG = None
from datetime import datetime, timedelta
import threading
import importlib.util
from chat_service import get_chat_service

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# In-memory processing lock to prevent duplicate processing of the same area while server is running
processing_jobs_in_memory = {}

# Firebase configuration
FIREBASE_CRED_PATH = "firebase-credentials.json"
FIRESTORE_COLLECTION = "building_damage_pins"
# Allow overriding the storage bucket via environment variable FIREBASE_STORAGE_BUCKET
# Also accept STORAGE_BUCKET as a fallback (some setups use that name)
# Priority: FIREBASE_STORAGE_BUCKET > STORAGE_BUCKET > config.py FIREBASE_CONFIG['storageBucket'] > hardcoded default
env_bucket = os.environ.get('FIREBASE_STORAGE_BUCKET')
if not env_bucket:
    # fallback to STORAGE_BUCKET if present
    env_bucket = os.environ.get('STORAGE_BUCKET')
if env_bucket:
    _raw_bucket = env_bucket
elif FIREBASE_CONFIG and isinstance(FIREBASE_CONFIG, dict):
    _raw_bucket = FIREBASE_CONFIG.get('storageBucket') or "jamaica-realtime-crisis-map.appspot.com"
else:
    _raw_bucket = "jamaica-realtime-crisis-map.appspot.com"
# Normalize bucket value: accept forms like 'gs://bucket-name' or full URLs and strip prefixes
if isinstance(_raw_bucket, str):
    STORAGE_BUCKET = _raw_bucket.strip()
    if STORAGE_BUCKET.startswith('gs://'):
        STORAGE_BUCKET = STORAGE_BUCKET[5:]
    if STORAGE_BUCKET.startswith('https://'):
        # e.g. https://storage.googleapis.com/bucket
        STORAGE_BUCKET = STORAGE_BUCKET.split('/')[-1]
    STORAGE_BUCKET = STORAGE_BUCKET.strip('/')
else:
    STORAGE_BUCKET = _raw_bucket

print(f"[CONFIG] env FIREBASE_STORAGE_BUCKET={os.environ.get('FIREBASE_STORAGE_BUCKET')}; env STORAGE_BUCKET={os.environ.get('STORAGE_BUCKET')}; computed STORAGE_BUCKET={STORAGE_BUCKET}")

# Initialize Firebase
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        # Try to derive default bucket from credentials project_id if possible
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
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_name_to_use})
            except Exception:
                firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app(cred)

    db = firestore.client()
    try:
        bucket = storage.bucket()
        # Verify that the bucket actually exists to avoid upload 404s later
        try:
            if hasattr(bucket, 'exists'):
                exists = bucket.exists()
            else:
                # Fallback: attempt to list one blob (will raise if bucket missing)
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
    print("✓ Firebase initialized successfully")
    if bucket is not None:
        print(f"✓ Storage bucket: {bucket.name}")
    else:
        print("⚠ No accessible storage bucket configured. Set STORAGE_BUCKET to a valid bucket name.")
except Exception as e:
    print(f"⚠ Firebase initialization failed: {e}")
    db = None
    bucket = None


@app.route('/')
def index():
    """Render main page with chat interface and map"""
    return render_template('index.html')


@app.route('/api/_debug/firebase', methods=['GET'])
def debug_firebase():
    """Debug endpoint returning Firebase config and storage bucket status"""
    info = {
        'env_FIREBASE_STORAGE_BUCKET': os.environ.get('FIREBASE_STORAGE_BUCKET'),
        'computed_STORAGE_BUCKET': STORAGE_BUCKET if 'STORAGE_BUCKET' in globals() else None,
        'firebase_app_initialized': bool(firebase_admin._apps),
        'firestore_connected': db is not None,
    }

    try:
        # Check bucket accessibility quickly
        b = None
        try:
            b = storage.bucket()
            # Attempt a light call to confirm access
            exists = False
            if b is not None:
                if hasattr(b, 'exists'):
                    exists = b.exists()
                else:
                    # attempt to list one blob
                    try:
                        next(b.list_blobs(max_results=1))
                        exists = True
                    except StopIteration:
                        exists = True
                    except Exception:
                        exists = False
        except Exception as e:
            info['bucket_check_error'] = str(e)
            exists = False

        info['bucket_accessible'] = bool(b) and exists
        if b is not None and hasattr(b, 'name'):
            info['bucket_name'] = b.name
    except Exception as e:
        info['error'] = str(e)

    return jsonify(info)


@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Fetch all detections from Firestore for map display"""
    if db is None:
        return jsonify({'error': 'Firestore not initialized'}), 500
    
    try:
        # Query parameters for filtering
        limit = request.args.get('limit', 1000, type=int)
        label_filter = request.args.get('label', None)
        
        # Build query
        query = db.collection(FIRESTORE_COLLECTION).limit(limit)
        
        if label_filter:
            query = query.where('properties.label', '==', label_filter)
        
        # Fetch documents
        docs = query.stream()
        
        features = []
        for doc in docs:
            data = doc.to_dict()
            # Ensure it's a valid GeoJSON feature
            if 'geometry' in data and 'properties' in data:
                feature = {
                    'type': 'Feature',
                    'id': doc.id,
                    'geometry': data['geometry'],
                    'properties': data['properties']
                }
                features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        return jsonify(geojson)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classifications', methods=['GET'])
def get_classifications():
    """Get LADI classification data for heatmap visualization"""
    if db is None:
        return jsonify({'error': 'Firestore not initialized'}), 500
    
    try:
        # Query parameters
        limit = request.args.get('limit', 5000, type=int)
        label_filter = request.args.get('label', None)
        
        # Query image_classifications collection
        collection_ref = db.collection('image_classifications')
        # If limit <= 0 or not provided, stream the entire collection (no limit)
        if limit is None or limit <= 0:
            docs = collection_ref.stream()
        else:
            docs = collection_ref.limit(limit).stream()
        
        # Group tiles by classification label
        classifications_by_label = {}
        all_tiles = []
        
        for doc in docs:
            data = doc.to_dict()
            
            # Extract tile info
            tile_info = {
                'tile': data.get('properties', {}).get('tile'),
                'tile_x': data.get('properties', {}).get('tile_x'),
                'tile_y': data.get('properties', {}).get('tile_y'),
                'zoom_level': data.get('properties', {}).get('zoom_level', 20),
                'coordinates': data.get('geometry', {}).get('coordinates'),
                'primary_label': data.get('properties', {}).get('primary_label'),
                'primary_confidence': data.get('properties', {}).get('primary_confidence'),
                'significant_labels': data.get('properties', {}).get('significant_labels', []),
                'all_predictions': data.get('properties', {}).get('all_predictions', [])
            }
            
            all_tiles.append(tile_info)
            
            # Group by primary label
            primary_label = tile_info['primary_label']
            if primary_label:
                if primary_label not in classifications_by_label:
                    classifications_by_label[primary_label] = []
                classifications_by_label[primary_label].append(tile_info)
            
            # Also group by significant labels (for multi-label support)
            for sig_label in tile_info['significant_labels']:
                if sig_label not in classifications_by_label:
                    classifications_by_label[sig_label] = []
                if tile_info not in classifications_by_label[sig_label]:
                    classifications_by_label[sig_label].append(tile_info)
        
        result = {
            'total_tiles': len(all_tiles),
            'by_label': classifications_by_label,
            'labels': list(classifications_by_label.keys())
        }
        
        # If label filter specified, return only that label's tiles
        if label_filter and label_filter in classifications_by_label:
            result['filtered_tiles'] = classifications_by_label[label_filter]
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for answering questions about LADI classification data
    Uses OpenAI GPT-4 with Firestore context
    """
    print(f"[CHAT] Received chat request")
    
    if db is None:
        print(f"[CHAT] ERROR: Firestore not initialized")
        return jsonify({'error': 'Firestore not initialized'}), 500
    
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        print(f"[CHAT] User message: {user_message}")
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get or create session ID for conversation history
        if 'chat_session_id' not in session:
            session['chat_session_id'] = str(uuid.uuid4())
        
        session_id = session['chat_session_id']
        print(f"[CHAT] Session ID: {session_id}")
        
        # Get chat service instance
        chat_service = get_chat_service(db)
        print(f"[CHAT] Calling chat service...")
        
        # Get response from GPT-4 with Firestore context
        response = chat_service.get_response(user_message, session_id)
        print(f"[CHAT] Got response: {response[:100]}...")
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'error': 'Failed to process chat message',
            'details': str(e)
        }), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history for current session"""
    try:
        session_id = session.get('chat_session_id')
        if session_id:
            chat_service = get_chat_service(db)
            chat_service.clear_history(session_id)
        
        # Create new session ID
        session['chat_session_id'] = str(uuid.uuid4())
        
        return jsonify({'status': 'success', 'message': 'Chat history cleared'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about detections"""
    if db is None:
        return jsonify({'error': 'Firestore not initialized'}), 500
    
    try:
        docs = db.collection(FIRESTORE_COLLECTION).stream()
        
        stats = {
            'total_count': 0,
            'by_label': {},
            'by_zoom': {}
        }
        
        for doc in docs:
            data = doc.to_dict()
            stats['total_count'] += 1
            
            # Count by label
            label = data.get('properties', {}).get('label', 'unknown')
            stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
            
            # Count by zoom level
            zoom = data.get('properties', {}).get('zoom_level', 'unknown')
            stats['by_zoom'][str(zoom)] = stats['by_zoom'].get(str(zoom), 0) + 1
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/storage_status', methods=['GET'])
def storage_status():
    """Return storage bucket name and accessibility status"""
    info = {
        'configured_bucket': STORAGE_BUCKET,
        'accessible': False,
        'bucket_name': None,
        'error': None
    }
    if db is None:
        info['error'] = 'Firestore not initialized'
        return jsonify(info), 200

    try:
        if 'bucket' in globals() and bucket is not None:
            info['accessible'] = True
            info['bucket_name'] = bucket.name
        else:
            # Try to initialize bucket temporarily to test access
            try:
                test_bucket = storage.bucket()
                info['accessible'] = True
                info['bucket_name'] = test_bucket.name
            except Exception as e:
                info['error'] = str(e)
                info['accessible'] = False
    except Exception as e:
        info['error'] = str(e)

    return jsonify(info), 200


@app.route('/api/process', methods=['POST'])
def process_area():
    """Trigger inference processing for a new area (runs in background)"""
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    # Only run LADI model
    models = ['ladi']
    
    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon are required'}), 400
    
    # Validate coordinates
    try:
        lat = float(lat)
        lon = float(lon)
        
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Latitude must be between -90 and 90'}), 400
        if not (-180 <= lon <= 180):
            return jsonify({'error': 'Longitude must be between -180 and 180'}), 400
            
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid lat/lon format'}), 400
    
    # Extract force flag and create a job key (rounded coordinates) to deduplicate nearby requests
    force = bool(data.get('force', False))
    key = f"lat:{round(lat,5)}_lon:{round(lon,5)}"

    # If job already running in-memory, reject immediately
    if key in processing_jobs_in_memory:
        return jsonify({'status': 'already_processing', 'message': 'This area is already being processed.'}), 409

    # Check Firestore for a recent job with same key (last 24h)
    try:
        if db is not None and not force:
            coll = db.collection('processing_jobs')
            docs = list(coll.where('job_key', '==', key).limit(1).stream())
            if docs:
                doc = docs[0]
                job_data = doc.to_dict()
                status = job_data.get('status')
                started_at = job_data.get('started_at')
                if status == 'running':
                    return jsonify({'status': 'already_processing', 'message': 'A processing job is already running for this area.'}), 409
                # If completed within 24 hours, consider it already processed
                try:
                    if started_at:
                        # started_at is a Firestore timestamp; compare to now using module-level datetime/timedelta
                        if isinstance(started_at, datetime):
                            if datetime.utcnow() - started_at <= timedelta(hours=24):
                                return jsonify({'status': 'already_processed', 'message': 'This area was processed recently.'}), 409
                except Exception:
                    pass
    except Exception:
        # Firestore may be unavailable; continue but rely on in-memory lock
        pass

    # Create a job document in Firestore so we can return a job_id to the client
    job_ref = None
    try:
        if db is not None:
            coll = db.collection('processing_jobs')
            job_doc = {
                'job_key': key,
                'lat': lat,
                'lon': lon,
                'status': 'queued',
                'created_at': datetime.utcnow()
            }
            added = coll.add(job_doc)
            job_ref = added[0] if isinstance(added, (list, tuple)) else added
            job_id = job_ref.id
        else:
            job_id = None
    except Exception as e:
        print(f"⚠ Could not create job doc: {e}")
        job_ref = None
        job_id = None

    # Define the background worker that actually runs the pipelines
    def run_inference_background():
        job_doc_ref = None
        try:
            # Mark in-memory
            processing_jobs_in_memory[key] = {'lat': lat, 'lon': lon, 'started': datetime.utcnow().isoformat(), 'models': models}
            # Update job doc to 'running' if we created one
            try:
                if db is not None and job_ref is not None:
                    job_ref.update({'status': 'running', 'started_at': datetime.utcnow(), 'models': models})
                    job_doc_ref = job_ref
            except Exception as e:
                print(f"⚠ Could not update job doc to running: {e}")

            print(f"🚀 Starting background inference for lat={lat}, lon={lon}, models={models}")

            # Run RescueNet if selected
            if 'rescuenet' in models:
                try:
                    spec = importlib.util.spec_from_file_location(
                        "rescue_mod", 
                        "rescuenet_yolo8_inference.py"
                    )
                    rescue = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(rescue)
                    # pass force flag so RescueNet can bypass per-tile dedupe if requested
                    if hasattr(rescue, 'process_zoom20_from_zoom15'):
                        try:
                            rescue_result = rescue.process_zoom20_from_zoom15(lat=lat, lon=lon, force=force)
                        except TypeError:
                            # older signature without force
                            rescue_result = rescue.process_zoom20_from_zoom15(lat=lat, lon=lon)
                    print(f"✅ RescueNet complete: {rescue_result}")
                except Exception as e:
                    print(f"❌ RescueNet error: {e}")

            # Run LADI classification if selected
            if 'ladi' in models:
                try:
                    spec_ladi = importlib.util.spec_from_file_location(
                        "ladi_mod",
                        "ladi_classification_inference.py"
                    )
                    ladi = importlib.util.module_from_spec(spec_ladi)
                    spec_ladi.loader.exec_module(ladi)
                    if hasattr(ladi, 'process_zoom20_from_zoom15'):
                        try:
                            ladi_result = ladi.process_zoom20_from_zoom15(lat=lat, lon=lon, force=force)
                        except TypeError:
                            ladi_result = ladi.process_zoom20_from_zoom15(lat=lat, lon=lon)
                    print(f"✅ LADI classification complete: {ladi_result}")
                except Exception as e:
                    print(f"❌ LADI classification error: {e}")

            # Run FloodNet if selected
            if 'floodnet' in models:
                try:
                    spec2 = importlib.util.spec_from_file_location(
                        "flood_mod",
                        "flooding_segmentation_inference.py"
                    )
                    flood = importlib.util.module_from_spec(spec2)
                    spec2.loader.exec_module(flood)
                    # pass force flag to FloodNet as well
                    if hasattr(flood, 'process_zoom20_from_zoom15'):
                        try:
                            flood_result = flood.process_zoom20_from_zoom15(lat=lat, lon=lon, force=force)
                        except TypeError:
                            flood_result = flood.process_zoom20_from_zoom15(lat=lat, lon=lon)
                    print(f"✅ FloodNet complete: processed {len(flood_result) if hasattr(flood_result, '__len__') else flood_result}")
                except Exception as e:
                    print(f"❌ FloodNet error: {e}")

            # Update job doc as completed
            try:
                if db is not None and job_doc_ref:
                    # job_doc_ref is a tuple (DocumentReference, write_time) returned by add()
                    doc_ref = job_doc_ref[0] if isinstance(job_doc_ref, (list, tuple)) else job_doc_ref
                    doc_ref.update({'status': 'completed', 'finished_at': datetime.utcnow()})
            except Exception as e:
                print(f"⚠ Could not update job doc: {e}")

            print("✅ Background inference finished for area")

        except Exception as e:
            print(f"❌ Background inference error: {e}")
            try:
                if db is not None and job_doc_ref:
                    doc_ref = job_doc_ref[0] if isinstance(job_doc_ref, (list, tuple)) else job_doc_ref
                    doc_ref.update({'status': 'failed', 'error': str(e), 'finished_at': datetime.utcnow()})
            except Exception:
                pass
        finally:
            # clear in-memory lock
            try:
                if key in processing_jobs_in_memory:
                    del processing_jobs_in_memory[key]
            except Exception:
                pass

    thread = threading.Thread(target=run_inference_background, daemon=True)
    thread.start()
    
    resp = {
        'status': 'processing',
        'message': f'Started processing area at ({lat}, {lon}) with models: {", ".join(models)}. Results will appear on the map when complete.',
        'lat': lat,
        'lon': lon,
        'job_id': job_id,
        'models': models
    }
    return jsonify(resp)


@app.route('/api/backfill_thumbnails', methods=['POST'])
def backfill_thumbnails():
    """Backfill thumbnails for existing Firestore detections.

    Request JSON options:
      - limit: int (max documents to process)
      - reprocess_all: bool (recreate thumbnails even if image_url exists)
      - label: optional label filter

    Runs in background and returns immediately.
    """
    if db is None:
        return jsonify({'error': 'Firestore not initialized'}), 500

    data = request.json or {}
    limit = int(data.get('limit', 500))
    reprocess_all = bool(data.get('reprocess_all', False))
    label_filter = data.get('label')

    def run_backfill():
        try:
            # Dynamically import the inference module which contains backfill_detection
            spec = importlib.util.spec_from_file_location('inference_module', 'rescuenet_yolo8_inference.py')
            inf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inf)

            query = db.collection(FIRESTORE_COLLECTION).limit(limit)
            if label_filter:
                query = query.where('properties.label', '==', label_filter)

            docs = list(query.stream())
            total = len(docs)
            processed = 0
            success = 0
            skipped = 0
            errors = []

            for doc in docs:
                doc_id = doc.id
                doc_dict = doc.to_dict()
                res = inf.backfill_detection(doc_id, doc_dict, overwrite=reprocess_all)
                processed += 1
                if res.get('ok'):
                    if res.get('skipped'):
                        skipped += 1
                    else:
                        success += 1
                else:
                    errors.append({'doc_id': doc_id, 'error': res.get('error')})

            print(f"✅ Backfill complete: processed={processed}, success={success}, skipped={skipped}, errors={len(errors)}")
            if errors:
                print('Sample errors:', errors[:5])
        except Exception as e:
            print(f"❌ Backfill failed: {e}")

    thread = threading.Thread(target=run_backfill, daemon=True)
    thread.start()

    return jsonify({'status': 'started', 'limit': limit, 'reprocess_all': reprocess_all, 'label': label_filter}), 202


@app.route('/api/job_status', methods=['GET'])
def job_status():
    """Return processing job status by job_id (query param)."""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({'error': 'job_id is required'}), 400
    if db is None:
        return jsonify({'error': 'Firestore not initialized'}), 500
    try:
        doc = db.collection('processing_jobs').document(job_id).get()
        if not doc.exists:
            return jsonify({'error': 'job not found'}), 404
        data = doc.to_dict()
        data['id'] = doc.id
        return jsonify({'status': 'ok', 'job': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask app...")
    print(f"Firestore collection: {FIRESTORE_COLLECTION}")
    import os
    print(f"Process PID: {os.getpid()}")
    # Run without the reloader to avoid multiple processes confusing local requests
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
