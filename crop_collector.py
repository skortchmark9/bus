import cv2

class CropCollector:
    def __init__(self, folder):
        # Search in folder for n existing crops
        self.folder = folder
        self.counter = len(sorted(folder.rglob("*.jpg")))
        print(f"Found {self.counter} existing crops in {folder}")

    def save(self, crop):
        # Save the crop to the folder with a unique name
        filename = f"{self.counter:06d}.jpg"
        filepath = self.folder / filename
        cv2.imwrite(str(filepath), crop)
        self.counter += 1
