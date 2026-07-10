import os
import subprocess
import zipfile
from pathlib import Path
from src.config import DATA_RAW, KAGGLE_USERNAME, KAGGLE_KEY

KAGGLE_EXE = r"C:\Users\AMAND\AppData\Local\Python\pythoncore-3.14-64\Scripts\kaggle.exe"


def _set_kaggle_env():
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY


def download_home_credit():
    _set_kaggle_env()
    out = DATA_RAW / "home-credit-default-risk.zip"
    if out.exists():
        print("Home Credit zip already downloaded.")
    else:
        print("Downloading Home Credit Default Risk dataset...")
        subprocess.run(
            [KAGGLE_EXE, "competitions", "download",
             "-c", "home-credit-default-risk", "-p", str(DATA_RAW)],
            check=True,
        )
        print("Download complete.")
    _unzip(out, DATA_RAW / "home_credit")


def download_lending_club():
    _set_kaggle_env()
    out = DATA_RAW / "lending-club.zip"
    if out.exists():
        print("Lending Club zip already downloaded.")
    else:
        print("Downloading Lending Club dataset...")
        subprocess.run(
            [KAGGLE_EXE, "datasets", "download",
             "-d", "wordsforthewise/lending-club", "-p", str(DATA_RAW)],
            check=True,
        )
        print("Download complete.")
    _unzip(out, DATA_RAW / "lending_club")


def _unzip(zip_path: Path, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"Already extracted: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"Extracted to {dest}")


if __name__ == "__main__":
    download_home_credit()
    download_lending_club()
