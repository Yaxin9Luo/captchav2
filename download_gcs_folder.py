import requests
import os
import sys

def download_gcs_folder(bucket_name, prefix, download_dir):
    """
    Downloads a folder from a public GCS bucket.
    """
    api_url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o"
    params = {"prefix": prefix}

    session = requests.Session()

    while True:
        try:
            response = session.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching file list: {e}", file=sys.stderr)
            return

        items = data.get("items", [])
        if not items and "nextPageToken" not in data:
            print("No files found in the specified folder.")
            break

        for item in items:
            file_path = item["name"]
            
            # The API returns the full prefix. We download relative to the prefix.
            relative_path = os.path.relpath(file_path, prefix)
            local_path = os.path.join(download_dir, relative_path)

            if file_path.endswith('/'):
                if not os.path.exists(local_path):
                    print(f"Creating directory {local_path}")
                    os.makedirs(local_path, exist_ok=True)
                continue

            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            download_url = f"https://storage.googleapis.com/{bucket_name}/{file_path}"
            
            # Check if file already exists and has the same size
            if os.path.exists(local_path) and os.path.getsize(local_path) == int(item['size']):
                print(f"Skipping {local_path} (already exists and size matches)")
                continue

            print(f"Downloading {download_url} to {local_path}")
            
            try:
                with session.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192*4):
                            f.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_path}: {e}", file=sys.stderr)
            except IOError as e:
                print(f"Error writing file {local_path}: {e}", file=sys.stderr)

        if "nextPageToken" in data:
            params["pageToken"] = data["nextPageToken"]
        else:
            break

    print("Download complete.")

if __name__ == "__main__":
    BUCKET_NAME = "physics-iq-benchmark"
    GCS_PREFIX = "full-videos/take-2/30FPS/"
    # The gsutil command would create 'full-videos' in the current directory.
    DOWNLOAD_DIRECTORY = "30fps_videos/take-2"
    
    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)

    download_gcs_folder(BUCKET_NAME, GCS_PREFIX, DOWNLOAD_DIRECTORY)
