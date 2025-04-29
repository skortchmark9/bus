import os
import json

output_dir = 'site'

def write_cameras(selected_cameras):
    with open(os.path.join(output_dir, 'cameras.json'), 'w') as f:
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
