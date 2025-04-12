from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
from datetime import datetime
import os

def extract_timestamp(filename):
    basename = os.path.basename(filename).replace(".jpg", "")
    return basename
    return datetime.fromtimestamp(int(basename))

def main():
    model = YOLO('models/yolov8n.pt')
    tracker = sv.ByteTrack()


    results = model.predict(
        source='data/raw',
        project='runs/buses',
        name='v0',
        save=True,
        classes=[5],  # class 5 = bus
    )

    bus_history = defaultdict(list)  # tracks {bus_id: [timestamps]}
    seen_buses = set()

    # Assume results[i] corresponds to images[i] (ordered)
    for result in results:
        timestamp = extract_timestamp(result.path)

        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id or []]

        for label in labels:
            bus_id = label[1:]  # Remove the '#' character
            bus_history[bus_id].append(timestamp)
            seen_buses.add(bus_id)

    for bus_id in sorted(bus_history.keys()):
        timestamps = sorted(bus_history[bus_id])
        if len(timestamps) < 2:
            continue
        appeared_at = timestamps[0]
        disappeared_at = timestamps[-1]

        print(f"🚌 Bus #{bus_id} appeared at {appeared_at} and left at {disappeared_at}")



if __name__ == '__main__':
    main()