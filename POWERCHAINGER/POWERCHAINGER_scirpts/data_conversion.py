from pathlib import Path
import os, sys, random


root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.config import ROOT_DIR
extra_dir = ROOT_DIR / "POWERCHAINGER" / "POWERCHAINGER_datasets" / "ExogeneData"
energy_dir = ROOT_DIR / "POWERCHAINGER" / "POWERCHAINGER_datasets" / "Inputs"
os.makedirs(extra_dir, exist_ok=True)
os.makedirs(energy_dir, exist_ok=True)

    
if __name__ == "__main__":
    os.chdir(ROOT_DIR / "POWERCHAINGER" / "POWERCHAINGER_datasets" / "users_raw")
    for csv_file in os.listdir():
        if csv_file.endswith(".csv"):
            df = pd.read_csv(csv_file, encoding='utf-16', skiprows=4, header=None, delim_whitespace=True)
            df.drop(columns=[0,1,2,3,4], inplace=True)
            df.columns = ["date", "id_pdl"]
            df["date"] = pd.to_datetime(df["date"]).apply(lambda x: x.replace(tzinfo=None))
            consumption_df = df[["id_pdl"]].astype(float)
            datetime_df = df[["date"]]
            del df
            datetime_df["hour"] = datetime_df["date"].dt.hour
            datetime_df["day_of_month"] = datetime_df["date"].dt.day

            datetime_df['minute'] = datetime_df['date'].dt.minute
            mask = datetime_df['minute'].isin([0, 30])
            final_df = datetime_df[mask].copy()
            consumption_df = consumption_df[mask].copy()
            final_df.drop(columns='minute', inplace=True)

            # Hours (24-hour cycle)
            final_df['hours_sin'] = np.sin(2 * np.pi * (final_df['hour']) / 24)
            final_df['hours_cos'] = np.cos(2 * np.pi * (final_df['hour']) / 24)
            final_df['days_sin'] = np.sin(2 * np.pi * (final_df['day_of_month']) / 31)
            final_df['days_cos'] = np.cos(2 * np.pi * (final_df['day_of_month']) / 31)
            final_df[["date", "hours_sin", "hours_cos", "days_sin", "days_cos"]].reset_index(drop=True).to_csv(os.path.join(extra_dir, csv_file), index=False, header=True)
            consumption_df = consumption_df/1000
            consumption_df.reset_index(drop=True, inplace=True)
            consumption_df = consumption_df.T.reset_index(drop=True)
            consumption_df["id_pdl"] = random.randint(1000,2000)
            col = consumption_df.pop("id_pdl")
            consumption_df.insert(0, 'id_pdl', col)
            del col
            consumption_df.to_csv(os.path.join(energy_dir, csv_file), index=False, header=True)
        else:
            continue