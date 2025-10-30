from multiprocessing import freeze_support, Pool, cpu_count
from pathlib import Path
import subprocess
import os
import glob
import shutil
import platform


def unzip_a_7z(args):
    zip_file, output_dir, sevenz_executable = args
    zip_file = Path(zip_file)

    if output_dir is None:
        output_dir = zip_file.parent / zip_file.stem.removesuffix(".7z")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        sevenz_executable,
        "x",
        str(zip_file),
        f"-o{output_dir}"
    ], check=True)

    print(f"Extracted {zip_file} to {output_dir}")

def autodetect_7z_executable(sevenz_executable):
    # --- Auto-detect OS and 7z path ---
    system = platform.system().lower()

    if system == "windows":
        default_7z_paths = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ]
    else:  # Linux / macOS
        default_7z_paths = ["7z", "7za"]

    # --- Determine which executable to use ---
    if sevenz_executable:
        sevenz_path = Path(sevenz_executable)
        if not sevenz_path.exists():
            raise FileNotFoundError(f"Specified 7z executable not found: {sevenz_executable}")
    else:
        # Try defaults and PATH
        sevenz_path = shutil.which("7z") or shutil.which("7za")
        if not sevenz_path:
            for candidate in default_7z_paths:
                if Path(candidate).exists():
                    sevenz_path = candidate
                    break

    # --- If still missing, raise clear error ---
    if not sevenz_path:
        raise FileNotFoundError(
            "7z executable not found.\n"
            "Please ensure 7-Zip is installed and added to PATH.\n"
            "Windows: https://www.7-zip.org/download.html\n"
            "Linux/macOS: install via package manager (e.g. `sudo apt install p7zip-full`)."
        )

    return sevenz_path

def unzip_dataset(dataset_folder, output_folder=None, multiprocess=False, sevenz_executable=None):
    """
    Extracts all .7z archives within the UrbanIng-V2X dataset folder using the 7-Zip command-line tool.

    This function searches for all `.7z` files inside `dataset_folder` (recursively or in the root,
    depending on your implementation), and extracts each archive into a corresponding subfolder.
    Optionally, it can use multiprocessing to parallelize the extraction across multiple CPU cores.

    Args:
        dataset_folder (str or Path):
            Path to the root folder containing one or more `.7z` archives of the UrbanIng-V2X dataset.

        output_folder (str or Path, optional):
            Directory where the extracted contents will be stored.
            If None (default), each archive is extracted into a subdirectory next to its `.7z` file.

        multiprocess (bool, optional):
            If True, extraction will run in parallel using a multiprocessing pool.
            The number of workers is automatically determined by available CPU cores.

        sevenz_executable (str or None, optional):
            Path to a custom `7z` or `7za` executable.
            If None, the function attempts to auto-detect the appropriate binary:
            - On Windows: tries common installation paths (e.g., `C:\\Program Files\\7-Zip\\7z.exe`)
            - On Linux/macOS: expects `7z` or `7za` to be available in the system PATH.

    Raises:
        FileNotFoundError:
            If the dataset folder does not exist, or if no `.7z` files are found, or if the 7-Zip executable is missing.

        RuntimeError:
            If extraction of any archive fails due to a subprocess or I/O error.

    Example:
        >>> unzip_dataset("datasets/UrbanIng-V2X", multiprocess=True)
    """

    freeze_support()

    if output_folder is None:
        output_folder = dataset_folder
    else:
        os.makedirs(output_folder, exist_ok=True)

        for root, dirs, files in os.walk(dataset_folder):
            for this_dir in dirs:
                os.makedirs(os.path.join(output_folder, this_dir), exist_ok=True)
            for this_file in files:
                if '.7z.' not in this_file:
                    source_path = os.path.join(root, this_file)
                    destination_path = os.path.join(output_folder, os.path.relpath(source_path, dataset_folder))
                    shutil.copy(source_path, destination_path)

    sevenz_executable = autodetect_7z_executable(sevenz_executable)

    args = [
        (sevenz, os.path.join(output_folder, os.path.relpath(sevenz, dataset_folder)).split('.')[0], sevenz_executable)
        for sevenz in sorted(glob.glob(os.path.join(dataset_folder, '**/*.7z.001'), recursive=True))
    ]

    if multiprocess:
        with Pool(processes=cpu_count()) as pool:
            pool.map(unzip_a_7z, args)

    else:
        for i, (zip_file, output_dir, this_executable_7z) in enumerate(args):
            print(f"Unzipping ({i+1} / {len(args)}): {zip_file} to {output_dir}")
            unzip_a_7z(args[i])
