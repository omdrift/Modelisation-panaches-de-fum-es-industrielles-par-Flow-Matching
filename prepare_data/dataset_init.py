import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor

JSON_FILE = 'metadata_02242020.json'  # Path to your metadata file
DOWNLOAD_DIR = 'smoke_videos'          # Folder where videos will be saved
RESOLUTION = 180                       # Choose 180 or 320
NUM_THREADS = 10                       # Number of simultaneous downloads
TARGET_LABELS = [23]               # 23 = Strong Positive (highest quality available in this JSON)
EXCLUDE_LABELS = [-2]              # -2 = Bad data
# ---------------------


def download_video(item):
    """Downloads a single video based on the metadata item."""
    label_state = item.get('label_state')
    label_admin = item.get('label_state_admin')
    
    # Skip bad data
    if label_state in EXCLUDE_LABELS or label_admin in EXCLUDE_LABELS:
        return
    
    # Download only Strong Positive videos (label 23)
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

    print(f"Total items in JSON: {len(data)}")
    
    # Debug: Show first item structure
    if data:
        print("\n--- Sample item structure ---")
        sample = data[0]
        print(f"Keys in item: {list(sample.keys())}")
        if 'label_state' in sample:
            print(f"label_state type: {type(sample['label_state'])}, value: {sample['label_state']}")
        if 'label_state_admin' in sample:
            print(f"label_state_admin type: {type(sample['label_state_admin'])}, value: {sample['label_state_admin']}")
    
    # Collect all unique labels to see what's available
    all_labels_state = set()
    all_labels_admin = set()
    for item in data:
        if 'label_state' in item and item['label_state'] is not None:
            all_labels_state.add(item['label_state'])
        if 'label_state_admin' in item and item['label_state_admin'] is not None:
            all_labels_admin.add(item['label_state_admin'])
    
    print(f"\nUnique label_state values: {sorted(all_labels_state)}")
    print(f"Unique label_state_admin values: {sorted(all_labels_admin)}")
    
    # Filter for Strong Positive (label 23 - highest quality in this JSON)
    print(f"\n🎯 Mode: Download STRONG POSITIVE videos (label {TARGET_LABELS})")
    to_download = [
        item for item in data 
        if item.get('label_state') in TARGET_LABELS 
        or item.get('label_state_admin') in TARGET_LABELS
    ]
    
    print(f"Found {len(to_download)} Strong Positive videos to download.")
    
    if len(to_download) == 0:
        print("\n⚠️  No videos found with the target labels!")
        print("💡 Suggestion: Check if the labels exist in the values above.")
        print("   Available labels:", sorted(all_labels_state | all_labels_admin))
        return
    
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