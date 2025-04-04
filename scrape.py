import asyncio
import json
import asyncio
import aiohttp
import aiofiles
import os
from datetime import datetime

cameras_path = 'data/cameras_manhattan.json'


def load_cameras():
    with open(cameras_path, 'r') as file:
        cameras = json.load(file)

    return cameras



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
    while True:
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
