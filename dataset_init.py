import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor

JSON_FILE = 'metadata_02242020.json'  # Path to your metadata file
DOWNLOAD_DIR = 'smoke_videos'          # Folder where videos will be saved
RESOLUTION = 180                       # Choose 180 or 320
NUM_THREADS = 10                       # Number of simultaneous downloads
TARGET_LABELS = [23, 47]               # Strong Positive and Gold Standard Positive
# ---------------------

def download_video(item):
    """Downloads a single video based on the metadata item."""
    label_state = item.get('label_state')
    label_admin = item.get('label_state_admin')
    
    if label_state not in TARGET_LABELS and label_admin not in TARGET_LABELS:
        return

    # Construct the URL
    url_root = item['url_root']
    url_part = item['url_part']
    
    if RESOLUTION == 320:
        url_root = url_root.replace('/180/', '/320/')
        url_part = url_part.replace('-180-180-', '-320-320-')
    
    url = url_root + url_part
    file_name = f"{item['id']}_{os.path.basename(url_part)}"
    file_path = os.path.join(DOWNLOAD_DIR, file_name)

    if os.path.exists(file_path):
        return f"Skipped: {file_name} (Already exists)"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return f"Downloaded: {file_name}"
    except Exception as e:
        return f"Failed: {file_name} - {e}"

def main():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    print(f"Loading {JSON_FILE}...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    to_download = [
        item for item in data 
        if item.get('label_state') in TARGET_LABELS 
        or item.get('label_state_admin') in TARGET_LABELS
    ]
    
    print(f"Found {len(to_download)} videos matching labels {TARGET_LABELS}.")
    print(f"Starting download to '{DOWNLOAD_DIR}' using {NUM_THREADS} threads...")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(download_video, to_download))

    success = [r for r in results if r and r.startswith("Downloaded")]
    skipped = [r for r in results if r and r.startswith("Skipped")]
    failed = [r for r in results if r and r.startswith("Failed")]
    
    print(f"\n--- Process Complete ---")
    print(f" downloaded: {len(success)}")
    print(f"Skipped : {len(skipped)}")
    print(f"Failed : {len(failed)}")

if __name__ == "__main__":
    main()