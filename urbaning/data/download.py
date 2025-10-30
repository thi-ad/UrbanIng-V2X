import hashlib
import os
from pyDataverse.api import NativeApi


# ---------- Helper: compute checksum ----------
def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_dataset(download_dir):
    """
    Download the entire UrbanIng-V2X dataset and save it in the  download_dir folder.

    Args:
        download_dir (str or Path):
            Path to the root folder to download the UrbanIng-V2X dataset.
    """

    # ================== CONFIG ==================
    DATAVERSE_URL = "https://dataverse.harvard.edu"
    PERSISTENT_ID = "doi:10.7910/DVN/A9LPY7"
    # ===========================================

    # Initialize API
    api = NativeApi(DATAVERSE_URL)

    # ---------- Step 1: Get dataset JSON ----------
    dataset = api.get_dataset(PERSISTENT_ID).json()
    version_data = dataset['data']['latestVersion']
    files = version_data['files']

    print(f"Found {len(files)} files to download")

    # ---------- Step 2: Create local folder ----------
    os.makedirs(download_dir, exist_ok=True)

    # ---------- Step 3: Download each file ----------
    for i, f in enumerate(files):
        file_label = f['label']
        file_id = f['dataFile']['id']
        md5 = f['dataFile']['md5']

        directory_label = f.get('directoryLabel', "")
        output_dir = os.path.join(download_dir, directory_label)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, file_label)
        if os.path.isfile(output_path):
            if compute_md5(output_path) == md5:
                print(f"Skipping File {output_path} has already been downloaded")
                continue

        print(f"Downloading ({i + 1} / {len(files)}): {file_label}")

        download_url = f"{DATAVERSE_URL}/api/access/datafile/{file_id}"

        os.system(f'curl -L "{download_url}" -o "{output_path}"')

        print(f"Downloaded ({i + 1} / {len(files)}): {output_path}")

    print("\nAll files downloaded successfully.")


def download_one_sequence(download_dir, sequence_name="20241126_0017_crossing1_00"):
    """
    Download one sequence from UrbanIng-V2X dataset and save it in the download_dir folder.

    Args:
        download_dir (str or Path):
            Path to the root folder to download the UrbanIng-V2X dataset.
        sequence_name (str):
            Name of the sequence to download.
    """

    # ================== CONFIG ==================
    DATAVERSE_URL = "https://dataverse.harvard.edu"
    PERSISTENT_ID = "doi:10.7910/DVN/A9LPY7"
    # ===========================================

    # Initialize API
    api = NativeApi(DATAVERSE_URL)

    # ---------- Step 1: Get dataset JSON ----------
    dataset = api.get_dataset(PERSISTENT_ID).json()
    version_data = dataset['data']['latestVersion']
    files = version_data['files']

    # print(f"Found {len(files)} files to download")

    # ---------- Step 2: Create local folder ----------
    os.makedirs(download_dir, exist_ok=True)

    # ---------- Step 3: Download each file ----------
    for i, f in enumerate(files):
        file_label = f['label']


        if 'digital_twin' in file_label:
            continue

        if '202411' in file_label:
            if sequence_name not in file_label:
                continue

        file_id = f['dataFile']['id']
        md5 = f['dataFile']['md5']

        directory_label = f.get('directoryLabel', "")
        output_dir = os.path.join(download_dir, directory_label)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, file_label)
        if os.path.isfile(output_path):
            if compute_md5(output_path) == md5:
                print(f"Skipping File {output_path} has already been downloaded")
                continue

        print(f"Downloading : {file_label}")

        download_url = f"{DATAVERSE_URL}/api/access/datafile/{file_id}"

        os.system(f'curl -L "{download_url}" -o "{output_path}"')

        print(f"Downloaded : {output_path}")

    print("\nAll files downloaded successfully.")
