from pathlib import Path

BRATS_DATA_SOURCE_DIR = Path("/mnt/93E8-0534/JuanCarlos/BraTS2020")
BRATS_DATA_TRAIN_DIR = BRATS_DATA_SOURCE_DIR / "MICCAI_BraTS2020_TrainingData_ipynb"
BRATS_DATA_VAL_DIR = BRATS_DATA_SOURCE_DIR / "MICCAI_BraTS2020_ValidationData_ipynb"
BRATS_RESULTS_DIR = BRATS_DATA_SOURCE_DIR / "results"
PRETRAINED_MODELS_DIR = Path(
    "/mnt/93E8-0534/JuanCarlos/mri-classifcation-pretrained-models"
)
