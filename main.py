from bus_routes_at_location import (
    get_routes_for_camera,
    load_route_shapes,
    load_cameras,
)
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


def main():
    selected_routes = [
        'M101'
    ]
    cameras = load_cameras()
    route_shapes = load_route_shapes()
    selected_cameras = []

    for camera in cameras:
        routes = get_routes_for_camera(camera, route_shapes)
        if any(route in selected_routes for route in routes):
            selected_cameras.append(camera)

    sessions = []

    # Update this if you want to use a custom model
    model = CameraSession.default_yolo_model
    model = 'models/yolov8n.pt'
    # Tuned model
    model = 'models/best.pt'
    print("Using model:", model)

    for camera in selected_cameras:
        session = CameraSession(camera["id"], camera, model)
        sessions.append(session)

    for session in sessions:
        session.step()

        for track in session.bus_tracks.values():
            track.dump()
        

if __name__ == "__main__":
    main()
