from pathlib import Path
from framework.utils.losses import GeometricMask

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'datasets'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'

DATA_PARAMS = ['hours_cos', 'hours_sin', 'days_cos', 'days_sin']

TRAINING_PARAMS = {"input_dim": 1 + len(DATA_PARAMS), 
                      "window_size": 1024, 
                      "dim_model": 96, 
                      "batch_size": 16,
                      "epochs": 1, 
                      "lr": 1e-4, 
                      "wd": 1e-3,
                      "n_warmup_epochs": 0,
                      "p_es": 5,
                      "p_rlr": 3,
                      "GeoMask" : GeometricMask(mean_length=24, masking_ratio=0.5, 
                                                type_corrupt='zero', dim_masked=0)}

CASES = ["cooker_case", "dishwasher_case",
         "desktopcomputer_case", "tumbledryer_case",
         "tv_greater21inch_case", "tv_less21inch_case",
         "laptopcomputer_case", "waterheater_case"]

ADF_CASES = {"cooker_case": 0.754, "dishwasher_case": 0.738,
             "desktopcomputer_case": 0.612, "tumbledryer_case": 0.655,
             "tv_greater21inch_case": 0.59, "tv_less21inch_case": 0.543,
             "laptopcomputer_case": 0.652, "waterheater_case": 0.616}