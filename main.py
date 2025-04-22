from bus_routes_at_location import (
    get_routes_for_camera,
    load_route_shapes,
    load_cameras,
)
from runtime import CameraSession

def main():
    selected_routes = [
        'M104'
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

    for camera in selected_cameras:
        session = CameraSession(camera["id"], camera, model)
        sessions.append(session)

    for session in sessions:
        session.step()

        for track in session.bus_tracks.values():
            track.dump()
        

if __name__ == "__main__":
    main()
