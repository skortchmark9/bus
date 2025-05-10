import time
from collections import defaultdict
import shutil
from pathlib import Path
from bus_routes_at_location import (
    get_routes_for_camera,
    load_route_shapes,
    load_cameras,
)
from test_clip_finetune import RoutePredictor as ClipRoutePredictor
from resnet import RoutePredictor as ResnetRoutePredictor
from crop_collector import CropCollector
from runtime import CameraSession
import api

# {
#     "id": "1546f761-039c-4b5c-af5e-75c83c9f603f",
#     "name": "Lexington Ave @ 42 St",
#     "latitude": 40.7514494972593,
#     "longitude": -73.976060779701,
#     "area": "Manhattan",
#     "isOnline": "true",
#     "imageUrl": "https://webcams.nyctmc.org/api/cameras/1546f761-039c-4b5c-af5e-75c83c9f603f/image"
# },
# In [6]: get_routes_for_camera(cams[12], load_route_shapes())
# Out[6]: ['M101', 'M102', 'M103', 'M42']
#
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point


def plot_cameras_on_nyc_map(camera_list):
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(camera_list)
    gdf["geometry"] = gdf.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    gdf.set_crs(epsg=4326, inplace=True)  # WGS84 lat/lon
    gdf = gdf.to_crs(epsg=3857)  # Web Mercator for map tiles

    # Plot
    ax = gdf.plot(figsize=(10, 10), color="red", markersize=10)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)  # Clean NYC basemap
    plt.title("NYC Traffic Cameras")
    plt.show()




def main():
    t0 = time.time()
    selected_routes = [
        'M104',
    ]
    cameras = load_cameras()
    route_shapes = load_route_shapes()
    selected_route_shapes = { route: route_shapes[route] for route in selected_routes if route in route_shapes }
    api.write_routes(selected_route_shapes)
    selected_cameras = []

    for camera in cameras:
        routes = get_routes_for_camera(camera, selected_route_shapes)
        if any(route in selected_routes for route in routes):
            selected_cameras.append(camera)

    sessions = []

    # Make the cameras unique by id
    selected_cameras = {camera["id"]: camera for camera in selected_cameras}
    selected_cameras = list(selected_cameras.values())

    cameras_by_ave = defaultdict(list)
    for camera in selected_cameras:
        ave, st = camera['name'].split('@')
        cameras_by_ave[ave].append(st.strip())

    print("Cameras:")
    for ave, streets in cameras_by_ave.items():
        print(f"\t{ave} @ {', '.join(streets)}")


    t1 = time.time()

    # plot_cameras_on_nyc_map(selected_cameras)
    api.write_cameras(selected_cameras)


    # Update this if you want to use a custom model
    model = CameraSession.default_yolo_model
    model = 'models/yolov8n.pt'
    # Tuned model
    model = 'models/fine-tuned-50-epochs.pt'
    print("Using model:", model)

    route_predictor = ResnetRoutePredictor()
    # crop_collector = CropCollector(Path('data/examples/4_27_1500/crops_labelled'))
    crop_collector = None


    PHOTOS_DIR = 'data/examples/4_27_1500/camera_images_m104'
    output_dir = Path('site/output_50')
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    specific_camera_id = '45cb119b-0e4a-442e-9410-b810ab8c255d'
    specific_camera_id = None
    min_timestamp = None
    max_timestamp = None
    # min_timestamp = '20250428T234214'
    # max_timestamp = '20250428T234312'
    


    for camera in selected_cameras:
        if specific_camera_id and camera["id"] != specific_camera_id:
            continue
        image_folder = Path(PHOTOS_DIR, camera["id"])
        session = CameraSession(
            image_folder,
            output_dir,
            camera,
            model,
            route_predictor,
            crop_collector
        )
        sessions.append(session)

    i = 0
    while True:
        i += 1
        print('*' * 20)
        print(f"Iteration {i}")
        for session in sessions:
            session.step(min_timestamp, max_timestamp)
            api.write_buses(sessions)

        for session in sessions:
            session.dump_tracks()

        time.sleep(2)

if __name__ == "__main__":
    main()
