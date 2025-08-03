from framework.AD_Framework.Framework import TSDataset
from config.config import DATA_PARAMS, TRAINING_PARAMS
from utils.data_utils import CER_get_data_case

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def extract_probs(case, model, M, data_name, extra_name, scheme):
    ID = {}
    data_tuple = CER_get_data_case(case, exo_variable=DATA_PARAMS,
                                win=TRAINING_PARAMS['window_size'],
                                ratio_resample=0.8, data_name=data_name, extra_name=extra_name)
    if scheme == "validation":
        X_valid_voter = data_tuple[8]
        y_valid_voter = data_tuple[9]
        _voter = TSDataset(X_valid_voter, y_valid_voter)
        y = _voter[:][1].flatten()
    else:
        X_test_voter  = data_tuple[10]
        y_test_voter  = data_tuple[11]
        _voter = TSDataset(X_test_voter, y_test_voter)
        y = _voter[:][1].flatten()
    
    for i, inst in enumerate(_voter):
        inst_ts, inst_label = inst
        inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], M, inst_ts.shape[1]//M))
        n_obs_per_win = inst_ts.shape[-1] // TRAINING_PARAMS['window_size']
        inst_ts = inst_ts[:, :, :n_obs_per_win* TRAINING_PARAMS['window_size']]

        tmp = np.empty((n_obs_per_win, M,  TRAINING_PARAMS['window_size']))
        for im in range(M):
            tmp[:, im, :] = np.reshape(inst_ts[:, im, :], (n_obs_per_win,  TRAINING_PARAMS['window_size']))
        #Each house is shaped from (1, all_samples) to (number_of windows, features, window_size)
        inst_ts = tmp.astype(np.float32)
        del tmp
        #Repeating the label for every window and creating Pytorch time-series dataset
        rep_labels = np.repeat(inst_label, len(inst_ts))
        ts_dataset = TSDataset(inst_ts, np.repeat(inst_label, len(inst_ts)), scaler=True, scale_dim=[0])
        #Loading the data to DataLoader
        loader = torch.utils.data.DataLoader(ts_dataset, batch_size=1)
        with torch.no_grad():
            logits_proba = []
            for ts, _ in loader:
                model.eval()
                ts = Variable(ts.float()).to('cuda')
                logits = model(ts)
                logits = nn.Softmax(dim=1)(logits)
                if not logits_proba:
                    logits_proba = list(logits[:, 1].cpu().detach().numpy().ravel())
                else:
                    logits_proba = logits_proba + list(logits[:, 1].cpu().detach().numpy().ravel())
                    
            logits_proba = [float(x) for x in logits_proba]
            rep_labels = [float(x) for x in rep_labels]
            ID[i] = [logits_proba, rep_labels]
    return ID