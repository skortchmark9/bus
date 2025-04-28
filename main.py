import time
from pathlib import Path
from bus_routes_at_location import (
    get_routes_for_camera,
    load_route_shapes,
    load_cameras,
)
from test_clip_finetune import RoutePredictor
from runtime import CameraSession

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
        # 'M101',
        'M104',
    ]
    cameras = load_cameras()
    route_shapes = load_route_shapes()
    selected_cameras = []

    for camera in cameras:
        routes = get_routes_for_camera(camera, route_shapes)
        if any(route in selected_routes for route in routes):
            selected_cameras.append(camera)

    sessions = []
    print('\n'.join([camera["name"] for camera in selected_cameras]))
    # Make the cameras unique by id
    selected_cameras = {camera["id"]: camera for camera in selected_cameras}
    selected_cameras = list(selected_cameras.values())

    t1 = time.time()
    print(f"Time taken to filter cameras: {t1 - t0:.2f} seconds")

    plot_cameras_on_nyc_map(selected_cameras)


    # Update this if you want to use a custom model
    model = CameraSession.default_yolo_model
    model = 'models/yolov8n.pt'
    # Tuned model
    model = 'models/fine-tuned-50-epochs.pt'
    print("Using model:", model)

    route_predictor = RoutePredictor()

    PHOTOS_DIR = 'data/examples/4_27_1500/camera_images_m104'
    output_dir = 'output_50'

    for camera in selected_cameras:
        folder = Path(PHOTOS_DIR, camera["id"])
        session = CameraSession(folder, camera, model, route_predictor)
        sessions.append(session)

    min_timestamp = None
    max_timestamp = None
    # min_timestamp = '20250427T131710'
    # max_timestamp = '20250427T131900'
    for session in sessions:
        session.step(min_timestamp, max_timestamp)

        for track in session.bus_tracks.values():
            track.dump(session.camera_attributes)
        
    t2 = time.time()
    print(f"Time taken to process data: {t2 - t1:.2f} seconds")

if __name__ == "__main__":
    main()
