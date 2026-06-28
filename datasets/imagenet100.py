"""
Download the ImageNet-1K (ILSVRC 2012) dataset from image-net.org

Access to image-net.org:
    Register at https://image-net.org/download-images.php
    and accept the terms. Downloads only work for authorized accounts.
    If wget returns 401/403, try exporting cookies from your browser.
"""


import os
import sys
import hashlib
import subprocess
from pathlib import Path
import scipy.io
import shutil

# destination directory
OUTPUT_DIR = Path("./imagenet1k")

# extract tar archives after download
EXTRACT = True

# delete tar archives after extraction (to save disk space)
DELETE_TAR_AFTER_EXTRACT = False

# (Optional) If wget returns 401, export cookies from your browser using
# the "Get cookies.txt LOCALLY" extension and set the path here:
COOKIES_FILE = ""  # e.g. "/home/user/cookies.txt"

SYNSETS = [
    "n02869837",
    "n01749939",
    "n02488291",
    "n02107142",
    "n13037406",
    "n02091831",
    "n04517823",
    "n04589890",
    "n03062245",
    "n01773797",
    "n01735189",
    "n07831146",
    "n07753275",
    "n03085013",
    "n04485082",
    "n02105505",
    "n01983481",
    "n02788148",
    "n03530642",
    "n04435653",
    "n02086910",
    "n02859443",
    "n13040303",
    "n03594734",
    "n02085620",
    "n02099849",
    "n01558993",
    "n04493381",
    "n02109047",
    "n04111531",
    "n02877765",
    "n04429376",
    "n02009229",
    "n01978455",
    "n02106550",
    "n01820546",
    "n01692333",
    "n07714571",
    "n02974003",
    "n02114855",
    "n03785016",
    "n03764736",
    "n03775546",
    "n02087046",
    "n07836838",
    "n04099969",
    "n04592741",
    "n03891251",
    "n02701002",
    "n03379051",
    "n02259212",
    "n07715103",
    "n03947888",
    "n04026417",
    "n02326432",
    "n03637318",
    "n01980166",
    "n02113799",
    "n02086240",
    "n03903868",
    "n02483362",
    "n04127249",
    "n02089973",
    "n03017168",
    "n02093428",
    "n02804414",
    "n02396427",
    "n04418357",
    "n02172182",
    "n01729322",
    "n02113978",
    "n03787032",
    "n02089867",
    "n02119022",
    "n03777754",
    "n04238763",
    "n02231487",
    "n03032252",
    "n02138441",
    "n02104029",
    "n03837869",
    "n03494278",
    "n04136333",
    "n03794056",
    "n03492542",
    "n02018207",
    "n04067472",
    "n03930630",
    "n03584829",
    "n02123045",
    "n04229816",
    "n02100583",
    "n03642806",
    "n04336792",
    "n03259280",
    "n02116738",
    "n02108089",
    "n03424325",
    "n01855672",
    "n02090622",  
]

URLS = {
    "devkit": {
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
        "md5": None,
        "size_gb": 0.003,
    },
    "val": {
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "md5": "29b22e2961454d5413ddabcf34fc5622",
        "size_gb": 6.3,
    },
    "train": {
        "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
        "md5": "1d675b47d978889d74fa0da5fadfb00e",
        "size_gb": 138.0,
    },
}

# ILSVRC2012_devkit_t12.tar.gz -> ILSVRC2012_devkit_t12/data/{meta.mat, ILSVRC2012_validation_ground_truth.txt}
DEVKIT_DATA_DIR = "data/imagenet1k/ILSVRC2012_devkit_t12/data"

def check_wget():
    """Check if wget is installed."""
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
        print("aria2c is installed.")
    except FileNotFoundError:
        print("Error: aria2c is not installed. Please install aria2c and try again.")
        print("Linux: sudo apt install aria2")
        sys.exit(1)

def check_disk_space(dest_dir, required_gb):
    """Check if there is enough disk space."""
    statvfs = os.statvfs(dest_dir)
    avail_gb = (statvfs.f_bavail * statvfs.f_frsize) / 1e9
    if avail_gb < required_gb:
        print(f"Error: Not enough disk space. Required: {required_gb:.2f} GB, Available: {avail_gb:.2f} GB")
        sys.exit(1)
    else:
        print(f"Disk space check passed. Required: {required_gb:.2f} GB, Available: {avail_gb:.2f} GB")


def verify_md5(filepath, expected_md5):
    """Verify the MD5 checksum of a file."""
    print(f"Verifying MD5 for {filepath}...")
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual == expected_md5:
        print("MD5 checksum is correct.")
        return True
    else:
        print(f"MD5 checksum is incorrect! Expected: {expected_md5}, Actual: {actual}")
        return False


def wget(url, dest):
    """Download a file using wget."""
    cmd = [
        "aria2c",
        "-x", "16",  
        "-s", "16",       
        "-k", "1M",        
        "--file-allocation=none",
        "-o", str(dest.name),
        "-d", str(dest.parent),
        url
    ]
    if COOKIES_FILE:
        cmd.extend(["--load-cookies", COOKIES_FILE])
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error downloading {url}")
        sys.exit(1)
    

def download(key, dest_dir):
    info = URLS[key]
    filename = info["url"].split("/")[-1]
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size / 1e9 >= info["size_gb"]*0.99:
        print(f"{filename} already exists. Skipping download.")
        return dest

    print(f"Downloading {filename} ({info['size_gb']:.2f} GB)...")
    wget(info["url"], dest)

    if info.get("md5"):
        if not verify_md5(dest, info["md5"]):
            print(f"Error: MD5 checksum failed for {filename}. Please delete the file and try again.")
            sys.exit(1)

    return dest


def extract_train_subset(tar_path, dest_dir):
    """Extract only the specified 100 synsets from the training tar archive."""
    train_dir = dest_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # build list of target tar files for the specified synsets
    targets = [f"{s}.tar" for s in SYNSETS]

    print(f"Extracting specified synsets to {train_dir}...")

    cmd = ["tar", "-xf", str(tar_path), "-C", str(train_dir)] + targets
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error extracting {tar_path}")
        sys.exit(1)

    # extract the specified synset tar files
    synset_tars = list(train_dir.glob("*.tar"))
    total = len(synset_tars)
    print(f"Extracting {total} synset tar files...")

    for i, st in enumerate(synset_tars, 1):
        synset_dir = train_dir / st.stem
        synset_dir.mkdir(exist_ok=True)
        subprocess.run(["tar", "-xf", str(st), "-C", str(synset_dir)], capture_output=True, check=True)
        st.unlink()  # delete the synset tar file after extraction
        if i % 10 == 0 or i == total:
            print(f"Extracted {i}/{total} synsets")

    print("Finished extracting training subset.")

    if DELETE_TAR_AFTER_EXTRACT:
        tar_path.unlink()  # delete the original training tar file to save space
        print(f"Deleted {tar_path} to save disk space.")

def extract_val(tar_path, dest_dir):
    """Extract the validation set (all 50k images)."""
    val_dir = dest_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting validation set to {val_dir}...")
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(val_dir)], check=True)
    print("Finished extracting validation set.")

    if DELETE_TAR_AFTER_EXTRACT:
        tar_path.unlink()  # delete the validation tar file to save space
        print(f"Deleted {tar_path} to save disk space.")

def extract_devkit(tar_path, dest_dir):
    """Extract the devkit (contains metadata and synset info)."""
    print(f"Extracting devkit...")
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(dest_dir)], check=True)
    print("Finished extracting devkit.")

def build_val_wnid_list(devkit_data_dir):
    """
    Read the devkit meta.mat + ground truth file and a return a list 
    index i gives the WNID (synset) of validation image i+1
    """
    meta_path = Path(devkit_data_dir) / "meta.mat"
    gt_path = Path(devkit_data_dir) / "ILSVRC2012_validation_ground_truth.txt"

    if not meta_path.exists() or not gt_path.exists():
        print(f"Error: devkit metadata not found in {devkit_data_dir}. "
              f"Make sure the devkit has been extracted.")
        sys.exit(1)

    meta = scipy.io.loadmat(meta_path, squeeze_me=True, struct_as_record=False)["synsets"]
    # maos 
    idx_to_wnid = {int(entry.ILSVRC2012_ID): str(entry.WNID) for entry in meta}

    with open(gt_path) as f:
        class_indices = [int(line.strip()) for line in f]

    return [idx_to_wnid[idx] for idx in class_indices]

def filter_val_subset(val_dir, devkit_data_dir, dest_dir, synsets):
    """
    input: already extracted, flat validation folder containing all 50000 ILSVRC2012_val_*.JPEG images 
    """
    val_dir = Path(val_dir)
    dest_dir = Path(dest_dir)
    synset_set = set(synsets)

    print("Reading devkit ground-truth labels...")
    wnids = build_val_wnid_list(devkit_data_dir)

    images = sorted(val_dir.glob("ILSVRC2012_val_*.JPEG"))
    if not images:
        # fall back to any image extension
        images = sorted(p for p in val_dir.iterdir() if p.is_file())

    if len(images) != len(wnids):
        print(f"Warning: found {len(images)} images in {val_dir} but "
              f"{len(wnids)} ground-truth labels. Results may be misaligned "
              f"if any images are missing/renamed.")
        
    dest_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for img_path, wnid in zip(images, wnids):
        if wnid in synset_set:
            out_dir = dest_dir / wnid
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / img_path.name 
            shutil.copy2(img_path, out_path)
            kept += 1

    print(f"Copied {kept}/{len(images)} val images into {dest_dir} "
          f"across {len(synset_set)} classes.")
    return dest_dir


def verify_structure(base):
    train_dir = base / "train"
    val_dir = base / "val"

    if train_dir.exists():
        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        print(f"Training set: {len(classes)} classes")
        if classes:
            first = classes[0]
            imgs = list(first.iterdir())
            print(f"Example class: {first.name}, number of images: {len(imgs)}")
    else:
        print("Training directory not found!")

    if val_dir.exists():
        imgs = list(val_dir.iterdir())
        print(f"Validation set: {len(imgs)} images")
    else:
        print("Validation directory not found!")


def main():
    print("Starting ImageNet-100 subset download and extraction...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_wget()

    check_disk_space(OUTPUT_DIR, 138 + 6.3 + 14 + 6.3 + 1)

    devkit_path = download("devkit", OUTPUT_DIR)
    val_path = download("val", OUTPUT_DIR)
    train_path = download("train", OUTPUT_DIR)

    if EXTRACT:
        extract_devkit(devkit_path, OUTPUT_DIR)
        extract_val(val_path, OUTPUT_DIR)
        extract_train_subset(train_path, OUTPUT_DIR)
        verify_structure(OUTPUT_DIR)

    print("All done! The ImageNet-100 subset is ready at:", OUTPUT_DIR)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "filter-val":
        # python imagenet100.py filter-val
        val_dir = "data" / OUTPUT_DIR / "val" 
        dest_dir =  "data" / OUTPUT_DIR / "val_filtered"
        filter_val_subset(val_dir, DEVKIT_DATA_DIR, dest_dir, SYNSETS)
    else:
        main()