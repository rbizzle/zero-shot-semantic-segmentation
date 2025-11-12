# Disaster Detection Dashboard

Flask web application with GPT-like chat interface and OpenStreetMap visualization for building damage detections.

## Features

- 🤖 **Chat Interface**: GPT-style chat to query detection data
- 🗺️ **Interactive Map**: OpenStreetMap with clustered markers for detections
- 📊 **Statistics Panel**: Real-time stats on detection counts and types
- 🔍 **Filtering**: Filter detections by type/label
- 🎨 **Color-Coded Markers**: Visual distinction between damage types

## Setup

### 1. Install Dependencies

```powershell
pip install -r flask-requirements.txt
```

### 2. Firebase Credentials

Place your Firebase credentials file as `firebase-credentials.json` in the root directory.

To get your credentials:
1. Go to Firebase Console → Project Settings
2. Navigate to Service Accounts
3. Click "Generate new private key"
4. Save as `firebase-credentials.json`

### 3. Run the Application

```powershell
python app.py
```

The app will be available at: `http://localhost:5000`

## Usage

### Chat Commands

Try these queries in the chat interface:
- "Show me all detections"
- "How many buildings were detected?"
- "What types of damage are there?"
- "Show statistics"

### Map Controls

- **🔄 Refresh**: Reload data from Firestore
- **📍 Center**: Fit all markers in view
- **Filter Dropdown**: Show only specific detection types

### Map Interactions

- Click on markers to see detection details
- Clustered markers expand on zoom
- Popup shows: label, confidence, coordinates, tile info

## API Endpoints

### `GET /`
Renders the main dashboard interface.

### `POST /api/chat`
Handles chat messages.
```json
{
  "message": "Your question here"
}
```

### `GET /api/detections`
Fetches detections from Firestore as GeoJSON.

Query parameters:
- `limit` (default: 1000) - Max number of detections
- `label` - Filter by detection label

Returns:
```json
{
  "type": "FeatureCollection",
  "features": [...]
}
```

### `GET /api/stats`
Returns detection statistics.

Response:
```json
{
  "total_count": 150,
  "by_label": {
    "damaged-building": 45,
    "road": 105
  },
  "by_zoom": {
    "19": 150
  }
}
```

## File Structure

```
├── app.py                      # Flask server
├── templates/
│   └── index.html             # Main HTML template
├── static/
│   ├── styles.css             # Styling
│   └── app.js                 # Frontend JavaScript
├── flask-requirements.txt     # Python dependencies
└── firebase-credentials.json  # Firebase credentials (create this)
```

## Technologies

- **Backend**: Flask, Firebase Admin SDK
- **Frontend**: Vanilla JavaScript, Leaflet.js, Leaflet.markercluster
- **Map**: OpenStreetMap tiles
- **Database**: Google Firestore

## Notes

- Map is centered on Jamaica (18.038°N, 77.860°W) by default
- Markers use color coding based on detection type
- Supports up to 5000 detections with clustering for performance
- Responsive design works on desktop and mobile
