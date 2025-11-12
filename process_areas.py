"""
Process multiple areas for RescueNet YOLO inference
Command-line tool to run inference on different geographic locations
"""

import sys
import argparse
from rescuenet_yolo8_inference import process_zoom20_from_zoom15, log

# Predefined areas of interest
AREAS = {
    'jamaica_melissa': {
        'lat': 18.0384131,
        'lon': -77.8600417,
        'name': 'Jamaica - Hurricane Melissa Area'
    },
    'jamaica_kingston': {
        'lat': 17.9712,
        'lon': -76.7936,
        'name': 'Kingston, Jamaica'
    },
    'florida_keys': {
        'lat': 24.5551,
        'lon': -81.7800,
        'name': 'Florida Keys'
    },
    'puerto_rico': {
        'lat': 18.2208,
        'lon': -66.5901,
        'name': 'San Juan, Puerto Rico'
    }
}


def process_area(area_key=None, lat=None, lon=None, name=None):
    """Process a single area by key or custom coordinates"""
    
    if area_key:
        if area_key not in AREAS:
            log(f"❌ Unknown area: {area_key}")
            log(f"Available areas: {', '.join(AREAS.keys())}")
            return
        
        area = AREAS[area_key]
        lat = area['lat']
        lon = area['lon']
        name = area['name']
        
        log(f"Processing predefined area: {name}")
    
    elif lat is not None and lon is not None:
        name = name or f"Custom location ({lat}, {lon})"
        log(f"Processing custom location: {name}")
    
    else:
        log("❌ Must provide either area_key or lat/lon coordinates")
        return
    
    # Run the inference pipeline
    total_count = process_zoom20_from_zoom15(lat=lat, lon=lon)
    log(f"✅ Completed {name}: {total_count} detections saved to Firestore")
    
    return total_count


def process_all_areas():
    """Process all predefined areas"""
    log("=" * 60)
    log("Processing ALL predefined areas")
    log("=" * 60)
    
    total_all = 0
    for key, area in AREAS.items():
        log(f"\n{'='*60}")
        log(f"Starting: {area['name']}")
        log(f"{'='*60}")
        
        count = process_area(area_key=key)
        total_all += count
        
        log(f"✅ {area['name']} complete: {count} detections")
    
    log(f"\n{'='*60}")
    log(f"🎉 ALL AREAS COMPLETE: {total_all} total detections")
    log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Run RescueNet YOLO inference on geographic areas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a predefined area
  python process_areas.py --area jamaica_melissa
  
  # Process all predefined areas
  python process_areas.py --all
  
  # Process custom coordinates
  python process_areas.py --lat 18.5 --lon -77.5 --name "Custom Area"
  
  # List available areas
  python process_areas.py --list

Available areas:
  """ + '\n  '.join([f"{k}: {v['name']}" for k, v in AREAS.items()])
    )
    
    parser.add_argument('--area', type=str, help='Process a predefined area by key')
    parser.add_argument('--all', action='store_true', help='Process all predefined areas')
    parser.add_argument('--lat', type=float, help='Custom latitude')
    parser.add_argument('--lon', type=float, help='Custom longitude')
    parser.add_argument('--name', type=str, help='Name for custom location')
    parser.add_argument('--list', action='store_true', help='List available areas and exit')
    
    args = parser.parse_args()
    
    # List areas and exit
    if args.list:
        print("\nAvailable predefined areas:")
        print("=" * 60)
        for key, area in AREAS.items():
            print(f"  {key:20s} - {area['name']}")
            print(f"                       ({area['lat']}, {area['lon']})")
        print("=" * 60)
        return
    
    # Process all areas
    if args.all:
        process_all_areas()
        return
    
    # Process single area
    if args.area:
        process_area(area_key=args.area)
        return
    
    # Process custom coordinates
    if args.lat is not None and args.lon is not None:
        process_area(lat=args.lat, lon=args.lon, name=args.name)
        return
    
    # No valid arguments
    parser.print_help()


if __name__ == "__main__":
    main()
