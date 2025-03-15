import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_kaggle_dataset(dataset, download_path, extract_path):
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {dataset} to {download_path}...")
    api.dataset_download_files(dataset, path=download_path, unzip=False)

    zip_file_path = os.path.join(download_path, f"{dataset.split('/')[-1]}.zip")
    print(f"Extracting {zip_file_path} to {extract_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"Removing {zip_file_path}...")
    os.remove(zip_file_path)
    print("Download and extraction complete.")

if __name__ == "__main__":
    dataset = 'mohamedlotfy50/wmt-2014-english-german'
    download_path = 'data/'
    extract_path = 'data/'

    os.makedirs(download_path, exist_ok=True)
    download_and_extract_kaggle_dataset(dataset, download_path, extract_path)
