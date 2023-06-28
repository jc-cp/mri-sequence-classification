from pathlib import Path

# BASE_DIR = directory where all of the csv. files of the dataset have been generated and inferred to
BASE_DIR = Path("/home/jc053/GIT/mri-sequence-classification/data_csv/long/")
DIR_NO_OPS = BASE_DIR / "curated_no_ops"
DIR_MIXED = BASE_DIR / "curated_nifti_data"
DIR_RADART = BASE_DIR / "curated_radart"
DIR_DGM = BASE_DIR / "curated_dgm"

OUTPUT_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/curated_dgm_filtered"
)
