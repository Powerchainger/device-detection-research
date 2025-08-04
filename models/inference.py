import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import joblib

from utils.data_utils import get_data_inference
from utils.feature_utils import create_data
from framework.AD_Framework.Framework import TSDataset
from framework.utils.TransApp_utils import get_model_inst
from config.config import TRAINING_PARAMS, CHECKPOINT_DIR, CASES, DATA_PARAMS

def run_inference(input_path=None, house_name=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = get_data_inference(input_path, house_name, exo_variable=DATA_PARAMS)
    results_df = pd.read_csv(CHECKPOINT_DIR / "case_best" / "results.csv")
    results_df = results_df.where(pd.notna(results_df), None)
    print(f"Testing existence of devices to the testing house.")
    for case in CASES:
        case_path = CHECKPOINT_DIR / 'case_headers' / f"{case}.pt"
        classifier_name = results_df[results_df['Case'] == case]["Classifier"].iloc[0]
        model = get_model_inst(TRAINING_PARAMS['input_dim'],
                                TRAINING_PARAMS['window_size'],
                                TRAINING_PARAMS['dim_model'],
                                path_select_core=case_path,
                                mode='classification',
                                device=device
                                )
        model.to(device)
        inf_data = TSDataset(data, labels=None, scaler=True, scale_dim=[0])
        inf_data_loader = torch.utils.data.DataLoader(inf_data, batch_size=1, shuffle=False)
        logits_proba = []
        model.eval()
        with torch.no_grad():
            for ts in inf_data_loader:
                ts = Variable(ts.float()).to(device)
                logits = model(ts)
                logits = torch.nn.Softmax(dim=1)(logits)
                logits_proba += list(logits[:, 1].cpu().detach().numpy().ravel())
        
        logits_proba = {0: logits_proba}
        method_tag = results_df[results_df['Case'] == case]["Method_Tag"].iloc[0]
        
        
        # === Handle VOTER inference ===
        if method_tag == 'voter':
            quantile = results_df[results_df['Case'] == case]["Quantile"].iloc[0]
            print(f"Using VOTER for case {case} with quantile {quantile}.")
            # Apply quantile voting across 25 window probabilities
            logits_array = np.array(logits_proba[0])
            house_pred = np.quantile(logits_array, q=quantile)
            final_pred = np.rint(house_pred)  

            is_present = "present" if final_pred == 1 else None
            if is_present:
                print(f"Device {case} is considered as {is_present} (via VOTER at q={quantile}).")
            else:
                print(f"Device {case} does not exist in the testing house (via VOTER at q={quantile}).")

        # === Handle ML classifier inference ===
        else:
            print(f"Using ML classifier {classifier_name} for case {case}.")
            if method_tag != 'raw':
                logits_data = create_data(logits_proba, filter=True).fillna(0)
                if method_tag == 'features':
                    model_path = CHECKPOINT_DIR / "trained_voters" / f"{case}_{classifier_name}.joblib"
                    clf = joblib.load(model_path)
                    prediction = clf.predict_proba(logits_data)
                else:
                    path = CHECKPOINT_DIR / "trained_voters" / f"{case}_{classifier_name}_features.csv"
                    f_df = pd.read_csv(path)["0"].tolist()
                    logits_data = logits_data[f_df]
                    model_path = CHECKPOINT_DIR / "trained_voters" / f"{case}_{classifier_name}.joblib"
                    clf = joblib.load(model_path)
                    prediction = clf.predict_proba(logits_data)
            else:
                logits_data = create_data(logits_proba, filter=False)
                model_path = CHECKPOINT_DIR / "trained_voters" / f"{case}_{classifier_name}.joblib"
                clf = joblib.load(model_path)
                prediction = clf.predict_proba(logits_data)

            is_present = "present" if prediction[0][1] > 0.5 else None
            if is_present:
                print(f"Device {case} is considered as {is_present} with confidence {prediction[0][1]:.2f}.")
            else:
                print(f"Device {case} does not exist in the testing house.")

if __name__ == "__main__":
    run_inference()