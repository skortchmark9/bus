import json
import pandas as pd
from collections import defaultdict
import pandas as pd
from shapely.geometry import LineString, Point
from collections import defaultdict

GTFS_PATH = "data/gtfs/gtfs_manhattan"


def load_route_shapes():
    shapes = pd.read_csv(f"{GTFS_PATH}/shapes.txt")
    trips = pd.read_csv(f"{GTFS_PATH}/trips.txt")
    routes = pd.read_csv(f"{GTFS_PATH}/routes.txt")

    shape_to_route = (
        trips.merge(routes, on="route_id")[["shape_id", "route_short_name"]]
        .drop_duplicates()
    )

    shape_groups = shapes.sort_values(
        ["shape_id", "shape_pt_sequence"]
    ).groupby("shape_id")

    shape_lines = {}
    for shape_id, group in shape_groups:
        points = list(zip(group["shape_pt_lon"], group["shape_pt_lat"]))
        if len(points) > 1:
            shape_lines[shape_id] = LineString(points)

    route_shapes = defaultdict(list)
    for _, row in shape_to_route.iterrows():
        shape_id = row["shape_id"]
        route = row["route_short_name"]
        if shape_id in shape_lines:
            route_shapes[route].append(shape_lines[shape_id])

    return route_shapes

def load_cameras():
    with open('./data/cameras_manhattan.json', 'r') as file:
        cameras = json.load(file)
    return cameras

def get_routes_passing_through(lat, lon, route_shapes, radius_m=50):
    pt = Point(lon, lat)
    radius_deg = radius_m / 111320  # ~1 deg ≈ 111.32 km
    matched_routes = set()

    for route, shapes in route_shapes.items():
        for line in shapes:
            if line.buffer(radius_deg).contains(pt):
                matched_routes.add(route)

    return sorted(matched_routes)

def get_routes_for_camera(camera, shapes):
    return get_routes_passing_through(camera["latitude"], camera["longitude"], shapes)
