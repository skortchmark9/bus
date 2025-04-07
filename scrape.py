import asyncio
import json
import asyncio
import aiohttp
import aiofiles
import os
from datetime import datetime
import shutil


cameras_path = 'data/cameras_m104.json'


cameras_m104 = set([
    "04e09ed5-2d97-4e29-8438-b87748850dbb",
    "053e8995-f8cb-4d02-a659-70ac7c7da5db",
    "0bcfbc92-d455-4f62-846a-32afbefa3b4b",
    "0c9a2836-c408-48d3-85c7-1977c33d9133",
    "19789499-03ac-45de-9401-a7a71ea60d1e",
    "43bb4857-e86e-4775-916e-6b26cedaf554",
    "45cb119b-0e4a-442e-9410-b810ab8c255d",
    "4850e464-1111-4b5d-a72d-f54a0e12a789",
    "4ebe49c8-3e68-43f7-b415-da3db9f72693",
    "4ebe49c8-3e68-43f7-b415-da3db9f72693",
    "566bce47-4390-4ff3-94ab-7a0ca2989163",
    "566bce47-4390-4ff3-94ab-7a0ca2989163",
    "5c505897-b475-4359-897d-b064bdb9feef",
    "769a2e94-5bbc-4a03-86a7-39d3a70213f7",
    "78a5c61d-9dce-4b52-b0da-bf7d22c1f501",
    "83bf2591-579d-415b-a0d4-fe39868b46d1",
    "85809312-60f2-4a52-a694-82628529c05a",
    "98b37dd5-bd64-4ace-add0-37f5135df600",
    "98b37dd5-bd64-4ace-add0-37f5135df600",
    "bca4a4b0-d73f-4937-9301-85ff8293bd94",
    "d78ebdc6-2211-479e-bda6-59d79db20258",
    "f0b924fe-0df6-4ac2-85a2-5d2b96d081d5",
    "f0b924fe-0df6-4ac2-85a2-5d2b96d081d5",
    "fc7db899-7a79-41f3-b9b1-b23ce64c5a5e",
    "fc7db899-7a79-41f3-b9b1-b23ce64c5a5e",
])



def load_cameras():
    with open(cameras_path, 'r') as file:
        cameras = json.load(file)

    return cameras

def check_disk_space(path="/"):
    total, used, free = shutil.disk_usage(path)
    return free // (2**30)


async def fetch_and_save_image(session, camera):
    camera_id = camera["id"]
    url = camera["imageUrl"]
    folder = os.path.join("data", "camera_images", camera_id)
    os.makedirs(folder, exist_ok=True)

    try:
        async with session.get(url) as response:
            if response.status == 200:
                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                filename = f"{timestamp}.jpg"
                filepath = os.path.join(folder, filename)

                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(await response.read())
                print(f"Saved image for camera {camera_id} at {timestamp}")
            else:
                print(f"Failed to fetch image for {camera_id}: HTTP {response.status}")
    except Exception as e:
        print(f"Error fetching image for {camera_id}: {e}")

async def poll_camera(session, camera):
    while check_disk_space() > 2:
        await fetch_and_save_image(session, camera)
        await asyncio.sleep(2)

async def main():
    cameras = load_cameras()
    print(f"Found {len(cameras)} cameras in Manhattan.")
    async with aiohttp.ClientSession() as session:
        tasks = [poll_camera(session, cam) for cam in cameras]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

def tag_cameras(cameras):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    for cam in cameras:
        if 'streetView' in cam:
            continue
        if not os.path.exists(f'data/camera_images_test/{cam['id']}.jpg'):
            cam['streetView'] = False
            continue

        try:
            img = mpimg.imread(f'data/camera_images_test/{cam['id']}.jpg')
        except (FileNotFoundError, OSError):
            print(f"File not found for camera {cam['id']}")
            cam['streetView'] = False
            continue

        plt.imshow(img)
        plt.axis('off')  # Hide axis
        plt.show(block=False)
        plt.pause(0.001)
        x = input("Use this cam? (y/n): ")
        if x == 'stop':
            break
        cam['streetView'] = x in ('y', 'k')
