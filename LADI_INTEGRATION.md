# LADI Classification Integration

## Summary
Added support for user-selectable models (RescueNet, LADI, FloodNet) with heatmap visualization for LADI multi-label classifications.

## Backend Changes (`app.py`)

### 1. Model Selection in `/api/process` Endpoint
- Added `models` parameter accepting list of models: `['rescuenet', 'ladi', 'floodnet']`
- Validates and defaults to `['rescuenet', 'ladi']` if not specified
- Background worker conditionally runs selected models

### 2. New `/api/classifications` Endpoint
- Returns LADI classification data from `image_classifications` collection
- Groups tiles by classification label
- Provides counts and tile coordinates for heatmap rendering
- Response format:
```json
{
  "total_tiles": 150,
  "by_label": {
    "flooding_any": [...tile_data],
    "buildings_affected_or_greater": [...tile_data]
  },
  "labels": ["flooding_any", "buildings_affected_or_greater", ...]
}
```

### 3. LADI Pipeline Integration
- Imports `ladi_classification_inference.py` in background worker
- Calls `process_zoom20_from_zoom15()` when 'ladi' is selected
- Stores results in Firestore `image_classifications` collection

## Frontend Changes

### 1. Model Selection UI (`templates/index.html`)
Added checkboxes for model selection:
- ☑ RescueNet (object detection)
- ☑ LADI (multi-label classification)
- ☐ FloodNet (flood segmentation)

### 2. Heatmap Controls
New section in legend:
- **Load Classifications** button - fetches LADI data
- Toggleable checkboxes for each classification type
- Color-coded squares showing classification colors

### 3. JavaScript Updates (`static/app.js`)

#### New Variables:
```javascript
let heatmapLayers = {};  // Stores active heatmap layer groups
let classificationsData = null;  // Cached classification data
const classificationColors = {
  'flooding_any': 'rgba(0, 119, 190, 0.6)',
  'buildings_affected_or_greater': 'rgba(231, 76, 60, 0.6)',
  // ... 12 disaster classifications
};
```

#### New Functions:
- `loadClassifications()` - Fetches classification data from API
- `populateHeatmapControls()` - Creates toggle checkboxes for each label
- `showHeatmapLayer(label)` - Renders colored tile rectangles for a classification
- `hideHeatmapLayer(label)` - Removes heatmap layer from map
- `tileToBounds(x, y, z)` - Converts tile coordinates to lat/lng bounds
- `tileToLatLng(x, y, z)` - Tile math for coordinate conversion

#### Updated Functions:
- `processArea()` - Now accepts `models` parameter, sends to API
- Process button click handler - Reads model checkboxes before processing

## Usage

### Processing Area with Model Selection
1. Check desired models (RescueNet, LADI, FloodNet)
2. Click "⚡ Process Area"
3. Backend runs selected models in parallel
4. Results appear in respective collections

### Viewing LADI Heatmaps
1. Click "Load Classifications" in legend
2. System fetches all classification tiles from Firestore
3. Checkboxes appear for each classification type with counts
4. Toggle checkboxes to show/hide colored tile overlays
5. Click tiles for detailed classification info

## Classification Colors
- **Flooding** - Blue shades (darker = structures)
- **Buildings Affected** - Red (damage severity)
- **Debris** - Orange
- **Roads Damage** - Purple
- **Trees Damage** - Green
- **Water/Bridges** - Light blue/gray

## Data Flow
```
User selects models → /api/process → Background worker
                                     ↓
                            Runs: RescueNet, LADI, FloodNet
                                     ↓
                            Stores in Firestore collections:
                            - building_damage_pins (RescueNet)
                            - image_classifications (LADI)
                            - flood_extents (FloodNet)
                                     ↓
Frontend loads data → /api/detections (pins)
                      /api/classifications (heatmaps)
                                     ↓
                            Map displays both:
                            - Clustered pins (RescueNet)
                            - Colored tile overlays (LADI)
```

## Testing
Test script: `test_ladi_inference.py`
- Validates endpoint connectivity
- Checks output structure compliance with RescueNet format
- Tests Firebase Storage uploads
- Verifies Firestore writes

## Configuration
HF Endpoint: `https://p6def67acw3ebe9z.us-east-1.aws.endpoints.huggingface.cloud`
Token: Configured in `ladi_classification_inference.py`
Firestore Collection: `image_classifications`
Storage Path: `classification_originals/z{zoom}/{x}_{y}.png`
