# Device Detection Pipeline

This repository implements a modular pipeline for device detection using user energy consumption data. The pipeline supports feature extraction, model training, and device classification, and is designed for easy updates and maintenance.

## Directory Structure

- `checkpoints/` — Pretrained model checkpoints, classification headers, and extracted data for voter training.
- `config/` — Configuration scripts for pipeline settings.
- `datasets/` — Contains datasets. Add your own datasets here; update helper functions in `data_utils.py` as needed.
- `features/` — Scripts for feature extraction and voter training.
- `framework/` — Contains the ADF framework, transformer model scripts, and framework utilities.
- `models/` — Model scripts and settings for unsupervised pretraining and supervised classification.
- `utils/` — Utility scripts for the pipeline.
- `main.py` — Main script to run the pipeline; supports running all or individual stages.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository-name>
   ```
3. (Optional but recommended) Create a virtual environment with your chosen name:
   ```bash
   python -m venv <your-env-name>
   ```
4. Activate your virtual environment:
   - On Windows:
     ```bash
     <your-env-name>\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source <your-env-name>/bin/activate
     ```
5. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Preprocessed CER dataset is recommended. Download from [Google Drive](https://drive.google.com/drive/folders/1QKyRkXv3lA7JHNFDuc_ZvvtrwLrA7oJk?usp=drive_link) and copy the `Inputs`, `Labels`, and `ExogeneData` folders into `datasets/`.

**Note:** Access to CER data from ISSDA is required. Please acknowledge the CER Smart Metering Project and ISSDA in any work.

## Usage

Run the pipeline using:
```bash
python main.py --<args>
```
You can configure and run individual stages by invoking `main.py` with different arguments. Inspect cmd arguments with `python main.py --help` for more details.

## External Dataset Structure

External datasets should follow the same directory structure as the default `datasets/` folder. Specifically, your dataset directory must contain subfolders named `Inputs`, `ExogeneData`, and `Labels`, each holding relevant `.csv` files. The `Inputs` folder should include time series data files (e.g., `x_residential_25728.csv`) with first column the index of the house in the `id_pdl` and then the `int-type` index that represents time steps or features (e.g.,`id_pdl`, `0`, `1`, `2`, ...). 

The `ExogeneData` folder should contain files (e.g., `extra_25728.csv`) with columns for external variables such as `hours_cos`, `hours_sin`, `days_cos`, and `days_sin`that can be used as temporal features. The first columns should be `date`, which is later set to index. The value type should be `string` and has to have this layout `yyyy-mm-dd hh:mm:ss` (e.g., `2009-07-15 00:00:00`) which later converted into `pandas.datetime`. The number of rows should match the number of columns on the `.csv` file in the `Inputs` folder, because every row in the `ExogeneData` file corresponds to a time step in the `Inputs` file.

The `Labels` folder should provide case label files (e.g., `cooker_case.csv`). This file should include two columns: the first column should be `id_pdl` and the second column should be `label` (e.g. `0` or `1`). The `id_pdl` should be the same as in the `Inputs` folder.

## Adding devices to pipeline

To add new devices to the pipeline, follow these steps:

1. **Add apropriate label file in the `Labels` folder**: Create a new `.csv` file with the device name (e.g., `new_device_case.csv`).

2. **Update the `CASES` variable in `config.py`**: Add the new device to the `CASES` list in `config.py`. This will ensure that the pipeline recognizes the new device during training and classification. the name of the `.csv` file should be the same as the device name in the `CASES` variable (e.g., `new_device_case.csv` for `new_device_case`).

3. **Run the pipeline using the `--new_device` argument**: 
```bash 
python main.py --new_device --input_path <path_to_your_dataset> --data_name <your_dataset_name>
```
This will create a new classification header for the new device, which can later be used for classification. `--input_path` should point to the folder containing the `Inputs`, `ExogeneData`, and `Labels` folders, and `--data_name` should be the name of your dataset (e.g., `my_dataset`).

## License

Specify your license here. I DON'T KNOW WHAT TO WRITE HERE.

## Acknowledgements

CER dataset provided by ISSDA (Irish Social Science Data Archive).  
Please acknowledge:  
“CER Smart Metering Project - Electricity Customer Behaviour Trial, 2009-2010” or  
“CER Smart Metering Project - Gas Customer Behaviour Trial, 2009-2010.”  
Accessed via the Irish Social Science Data Archive - www.ucd.ie/issda. 
“ADF & TransApp: A Transformer-Based Framework for Appliance Detection Using Smart Meter Consumption Series” by Adrien Petralia, Philippe Charpentier, and Themis Palpanas, arXiv preprint arXiv:2401.05381 (2023).
