# Processing New Areas - Complete Guide

## Architecture Overview

The system has **two independent components**:

### 1. 🔬 Inference Pipeline (Data Producer)
- **File**: `rescuenet_yolo8_inference.py` or `process_areas.py`
- **Purpose**: Analyzes satellite imagery and writes to Firestore
- **Input**: Geographic coordinates (lat/lon)
- **Output**: Detection data in Firestore + thumbnails in Storage

### 2. 🌐 Web Dashboard (Data Consumer)
- **File**: `app.py` (Flask server)
- **Purpose**: Displays existing detection data
- **Input**: Reads from Firestore database
- **Output**: Interactive map visualization

---

## How to Process New Areas

### Method 1: Command-Line Tool (Recommended for Batch Processing)

#### Process a Single Custom Location
```powershell
python process_areas.py --lat 18.5 --lon -77.5 --name "Custom Area"
```

#### Process Predefined Areas
```powershell
# List available areas
python process_areas.py --list

# Process specific area
python process_areas.py --area jamaica_kingston

# Process ALL predefined areas
python process_areas.py --all
```

#### Add Your Own Predefined Areas
Edit `process_areas.py` and add to the `AREAS` dictionary:
```python
AREAS = {
    'your_area_name': {
        'lat': 25.7617,
        'lon': -80.1918,
        'name': 'Miami, Florida'
    },
    # ... add more areas
}
```

---

### Method 2: Web Interface (Real-Time Processing)

#### Steps:
1. **Start the Flask app**:
   ```powershell
   python app.py
   ```

2. **Open browser**: Navigate to `http://localhost:5000`

3. **Navigate on map**: Pan/zoom to your area of interest

4. **Click "⚡ Process Area"** button
   - Confirms current map center coordinates
   - Starts background processing
   - Auto-refreshes after 2 minutes

5. **Monitor progress**:
   - Check terminal for processing logs
   - Refresh map to see new detections appear

---

### Method 3: Direct Script Editing (Quick One-Off)

Edit `rescuenet_yolo8_inference.py` bottom section:

```python
if __name__ == "__main__":
    # Change these coordinates to your area
    total_count = process_zoom20_from_zoom15(lat=YOUR_LAT, lon=YOUR_LON)
    log(f"Pipeline completed. Total detections: {total_count}")
```

Then run:
```powershell
python rescuenet_yolo8_inference.py
```

---

## Complete Workflow Example

### Scenario: Analyze Hurricane Damage in New Location

1. **Choose your coordinates**:
   - Use Google Maps to find center point
   - Example: Havana, Cuba (23.1136, -82.3666)

2. **Run inference**:
   ```powershell
   python process_areas.py --lat 23.1136 --lon -82.3666 --name "Havana Cuba"
   ```

3. **What happens**:
   ```
   [2025-11-09 10:30:00] Starting pipeline at lat=23.1136, lon=-82.3666
   [2025-11-09 10:30:00] Zoom-15 bounding box calculated
   [2025-11-09 10:30:00] Zoom-19 tile range: x=2345-2360, y=3456-3471, total=256
   [2025-11-09 10:30:05] Downloading tiles...
   [2025-11-09 10:35:20] Running YOLO detection...
   [2025-11-09 10:38:45] Uploading thumbnails to Storage...
   [2025-11-09 10:40:12] Saving to Firestore...
   [2025-11-09 10:40:30] ✓ All detections saved: 342 total
   ```

4. **View results**:
   - Open/refresh web dashboard
   - Click "🔄 Refresh" button
   - Pan to Havana area
   - See 342 new detection markers!

---

## Understanding the Data Flow

```
Geographic Coordinates (lat/lon)
         ↓
    Tile Calculator
         ↓
Zoom 15 → Zoom 19 Tiles (256 tiles)
         ↓
   Download from NOAA
         ↓
  RescueNet YOLO Model
         ↓
    Detections Found
         ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Crop & Upload      Create GeoJSON
  Thumbnail           Feature
    ↓                   ↓
Firebase Storage   Firebase Firestore
    │                   │
    └────────┬──────────┘
             ↓
       Web Dashboard
             ↓
    Interactive Map
```

---

## Key Files and Their Roles

| File | Purpose | When to Use |
|------|---------|-------------|
| `rescuenet_yolo8_inference.py` | Core inference engine | Import for custom scripts |
| `process_areas.py` | CLI tool for batch processing | Process multiple areas |
| `app.py` | Web dashboard server | View and trigger processing |
| `templates/index.html` | Frontend UI | Customize interface |
| `static/app.js` | Map interaction logic | Add features |

---

## FAQ

### Q: Can I process areas without running the web app?
**A:** Yes! Use `process_areas.py` or run `rescuenet_yolo8_inference.py` directly.

### Q: How long does processing take?
**A:** ~5-10 minutes for a Zoom 15 area (256 tiles at Zoom 19).

### Q: Can I process multiple areas at once?
**A:** Yes, use `python process_areas.py --all` for all predefined areas.

### Q: What if NOAA tiles don't exist for my area?
**A:** The script will skip missing tiles and process available ones.

### Q: Can I change the NOAA URL for different events?
**A:** Yes, edit `NOAA_URL` in `rescuenet_yolo8_inference.py`:
```python
NOAA_URL = "https://stormscdn.ngs.noaa.gov/YOUR_EVENT_ID"
```

### Q: How do I know what areas have been processed?
**A:** Check the web dashboard stats panel or query Firestore directly.

---

## Firestore Data Structure

Each detection is stored as:
```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [lon, lat]
  },
  "properties": {
    "label": "damaged-building",
    "confidence": 0.87,
    "tile": "19/2345/3456",
    "zoom_level": 19,
    "tile_x": 2345,
    "tile_y": 3456,
    "bbox": {"x1": 120, "y1": 85, "x2": 180, "y2": 145},
    "image_url": "https://storage.googleapis.com/.../thumbnail.jpg",
    "timestamp": "2025-11-09T10:40:30.123Z"
  }
}
```

---

## Advanced: API Usage

### Trigger Processing via HTTP
```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"lat": 18.5, "lon": -77.5}'
```

### Fetch Detections
```bash
curl http://localhost:5000/api/detections?limit=100&label=damaged-building
```

### Get Statistics
```bash
curl http://localhost:5000/api/stats
```

---

## Troubleshooting

### No detections appearing on map
1. Check Firestore in Firebase Console
2. Verify `FIRESTORE_COLLECTION` name matches in both files
3. Check browser console for errors
4. Click "🔄 Refresh" button

### Processing fails
1. Check Firebase credentials are valid
2. Verify NOAA URL is accessible
3. Check internet connection
4. Review terminal logs for specific errors

### Thumbnails not showing
1. Verify `STORAGE_BUCKET` is configured
2. Check Firebase Storage rules allow public read
3. Ensure images uploaded successfully (check Storage in Firebase Console)

---

## Next Steps

1. **Add more predefined areas** to `process_areas.py`
2. **Schedule regular processing** using cron/Task Scheduler
3. **Set up monitoring** for new NOAA imagery
4. **Customize detection types** in the web interface
5. **Export results** to GeoJSON/KML for GIS tools
