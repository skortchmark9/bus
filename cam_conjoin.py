import shutil
import re
from pathlib import Path

ROOT_DIR = Path("data")
MERGED_DIR = ROOT_DIR / "merged"
MERGED_DIR.mkdir(exist_ok=True)

# simple UUID regex
uuid_pattern = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)

for jpg_path in ROOT_DIR.rglob("*.jpg"):
    uuid_dir = jpg_path.parent

    # only process files whose immediate parent is a UUID
    if not uuid_pattern.match(uuid_dir.name):
        continue

    # now treat the UUID itself as the camera identifier
    camera_uuid = uuid_dir.name
    target_dir = MERGED_DIR / camera_uuid
    target_dir.mkdir(parents=True, exist_ok=True)

    dest = target_dir / jpg_path.name
    shutil.copy(str(jpg_path), str(dest))

print(f"Done. All images grouped by camera UUID under {MERGED_DIR}")
