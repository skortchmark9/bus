from pathlib import Path
from collections import defaultdict, deque
import cv2
from ultralytics import YOLO
import supervision as sv
from datetime import datetime, timedelta
from find_buses import is_mta_blue


PHOTOS_DIR = 'data/camera_images_old'

from itertools import count
from collections import defaultdict
from datetime import timedelta

class TrackerManager:
    def __init__(self, camera_name, gap_threshold_sec=5):
        self.camera_name = camera_name
        self.tracker = sv.ByteTrack()
        self.generation = 0
        self.last_timestamp = None
        self.gap_threshold = timedelta(seconds=gap_threshold_sec)
        self.id_counter = count(1)
        self.active_ids = defaultdict(dict)  # generation → tracker_id → bus_id

    def update(self, detections, timestamp):
        if self.last_timestamp and (timestamp - self.last_timestamp) > self.gap_threshold:
            print(f"[{self.camera_name}] Tracker reset due to time gap.")
            self.tracker = sv.ByteTrack()
            self.generation += 1
        self.last_timestamp = timestamp

        updated_dets = self.tracker.update_with_detections(detections)
        mapped_ids = []

        for i, tracker_id in enumerate(updated_dets.tracker_id):
            if tracker_id not in self.active_ids[self.generation]:
                bus_id = f"{self.camera_name}#{next(self.id_counter)}"
                self.active_ids[self.generation][tracker_id] = bus_id
            else:
                bus_id = self.active_ids[self.generation][tracker_id]
            mapped_ids.append(bus_id)

        return updated_dets.xyxy, mapped_ids


class BusTrack:
    def __init__(self, bus_id):
        self.bus_id = bus_id
        self.timestamps = []
        self.bboxes = []
        self.route_preds = []
        self.first_seen = None
        self.last_seen = None
        self.frames = []

    def update(self, timestamp, bbox, route_pred, frame):
        if not self.first_seen:
            self.first_seen = timestamp
        self.timestamps.append(timestamp)
        self.bboxes.append(bbox)
        self.route_preds.append(route_pred)
        self.last_seen = timestamp
        self.frames.append(frame)

    def is_expired(self, current_time, timeout=timedelta(seconds=10)):
        return self.last_seen and (current_time - self.last_seen) > timeout

    def get_final_route(self):
        counts = defaultdict(int)
        for r in self.route_preds:
            counts[r] += 1
        if not counts:
            return "Uncertain"
        return max(counts, key=counts.get)
    
    def dump(self):
        # Create output directory if it doesn't exist
        output_dir = Path("output", self.bus_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        for timestamp, bbox, frame in zip(self.timestamps, self.bboxes, self.frames):
            # Add bbox to frame and write to disk
            x1, y1, x2, y2 = bbox
            timestamp_string = timestamp.strftime("%Y%m%dT%H%M%S")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(f"output/{self.bus_id}/{timestamp_string}.jpg", frame)


    def __str__(self):
        route = self.get_final_route()
        n_frames = len(self.timestamps)
        return f"Bus {self.bus_id} {n_frames} frames seen from {self.first_seen} to {self.last_seen}, route: {route}"

class CameraSession:
    def __init__(self, camera_id, camera_attributes):
        self.camera_id = camera_id
        self.camera_attributes = camera_attributes
        self.folder = Path(PHOTOS_DIR, camera_id)
        self.last_seen = ""
        self.bus_tracks = {}
        self.tracker = TrackerManager(camera_name=self.camera_attributes["name"])
        self.model = YOLO("yolov8n.pt", verbose=False)


    def get_new_frames(self):
        all_frames = sorted(self.folder.glob("*.jpg"))
        new_frames = [f for f in all_frames if f.name > self.last_seen]
        if new_frames:
            self.last_seen = new_frames[-1].name
        return new_frames

    def process_frame(self, frame_path):
        frame = cv2.imread(str(frame_path))
        timestamp = frame_path.stem
        timestamp = datetime.strptime(timestamp, "%Y%m%dT%H%M%S")

        results = self.model.predict(frame_path, classes=[5], verbose=False)

        detections = sv.Detections.from_ultralytics(results[0])
        xyxys, bus_ids = self.tracker.update(detections, timestamp)
        for xyxy, bus_id in zip(xyxys, bus_ids):
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            is_blue = is_mta_blue(crop)
            if not is_blue:
                continue

            route_pred = self.recognize_sign(crop)

            if bus_id not in self.bus_tracks:
                self.bus_tracks[bus_id] = BusTrack(bus_id)

            self.bus_tracks[bus_id].update(timestamp, (x1, y1, x2, y2), route_pred, frame.copy())

    def flush_expired(self):
        now = datetime.now()
        expired_ids = [bid for bid, track in self.bus_tracks.items() if track.is_expired(now)]
        for bid in expired_ids:
            bus_track = self.bus_tracks[bid]
            print(bus_track)
            # del self.bus_tracks[bid]

    def recognize_sign(self, crop):
        # placeholder for OCR / CLIP / classifier logic
        return "Uncertain"

    def step(self):
        for frame in self.get_new_frames():
            self.process_frame(frame)
        self.flush_expired()
