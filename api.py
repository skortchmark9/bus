import os
import json

output_dir = 'site'

def write_cameras(selected_cameras):
    with open(os.path.join(output_dir, 'cameras.json'), 'w') as f:
        json.dump(selected_cameras, f, indent=4)
