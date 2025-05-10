import pandas as pd
import cv2

class CropCollector:
    def __init__(self, folder):
        # Search in folder for n existing crops
        self.folder = folder
        self.output_labels = 'data/crop_labels.csv'
        self.counter = len(sorted(folder.rglob("*.jpg")))
        print(f"Found {self.counter} existing crops in {folder}")

    def save(self, crop):
        # Save the crop to the folder with a unique name
        filename = f"{self.counter:06d}.jpg"
        filepath = self.folder / filename
        cv2.imwrite(str(filepath), crop)
        self.counter += 1
        return filename

    def import_crops(self, import_folder):
        import_labels = pd.read_excel(import_folder / 'RouteIdentification.xlsx', converters={'Name': str})
        # Conver to a name/label dict
        name_label_dict = {f'{row['Name']}.jpg': row['Route'] for _, row in import_labels.iterrows()}
        new_labels = {}

        # Import crops from the import folder
        for crop_path in sorted(import_folder.rglob("*.jpg")):
            crop_basename = crop_path.name
            label = name_label_dict.get(crop_basename)

            new_path = self.save(cv2.imread(str(crop_path)))  
            if not label:
                continue

            if label.startswith('M'):
                label = label.replace('M', 'm')
            new_labels[new_path] = label      

        with open(self.output_labels, "a") as f:
            for path, label in new_labels.items():
                f.write(f"{path},{label}\n")