import json
from pathlib import Path
import os
import json

output_dir = 'site'

def write_cameras(selected_cameras, filename="cameras.json"):
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(selected_cameras, f)


def write_routes(route_lines, filename="routes.json"):
    output = {}

    for route_id, shapes in route_lines.items():
        coords = []

        if not isinstance(shapes, list):
            shapes = [shapes]

        for shape in shapes:
            if hasattr(shape, 'coords'):  # It's a LineString
                coords.extend(list(shape.coords))
            else:
                coords.extend(shape)  # Assume already a list

        # Flip lon/lat -> lat/lon for Leaflet
        output[route_id] = [[lat, lon] for lon, lat in coords]

    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(output, f)


def write_buses(sessions, filename="buses.json"):
    output = {}

    for session in sessions:
        camera_id = session.camera_attributes["id"]

        for bus_id, track in session.bus_tracks.items():
            arrival = track.first_seen
            departure = track.last_seen
            image_path = track.best_frame_path().replace('site/', '')

            bus_info = {
                "bus_id": bus_id,
                "route": track.get_final_route(),
                "arrived": arrival.isoformat() + 'Z',
                "departed": departure.isoformat() + 'Z',
                "image": image_path
            }

            if camera_id not in output:
                output[camera_id] = []

            output[camera_id].append(bus_info)

    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(output, f)
