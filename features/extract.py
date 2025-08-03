from framework.utils.TransApp_utils import get_model_inst
from config.config import CASES, DATA_PARAMS, TRAINING_PARAMS, CHECKPOINT_DIR
from framework.utils.extract_probs import extract_probs
from utils.data_utils import create_dir

import json

M = 1 + len(DATA_PARAMS) # Number of modalities

def extract(data_name='x_residential_25728', extra_name='extra_25728', mode='validation'):
    if mode=='validation':
        save_path = str(CHECKPOINT_DIR) + "\\valid_probs"
    else:
        save_path = str(CHECKPOINT_DIR) + "\\test_probs"
    create_dir(save_path)
    for case in CASES:
        checkpoint_path = str(CHECKPOINT_DIR) + "\\case_headers" + f"\\{case}.pt"
        model = get_model_inst(M, TRAINING_PARAMS['window_size'],
                               TRAINING_PARAMS['dim_model'],
                               path_select_core=checkpoint_path,
                               mode='classification'
        )
        model.to("cuda")
        preds_dict = extract_probs(case, model, M, data_name, extra_name, scheme=mode)
        if mode == 'validation':
            file = save_path + f"\\{case}_valid_probs"
            with open(file + '.json', 'w') as json_file:
                json.dump(preds_dict, json_file)
        else:
            file = save_path + f"\\{case}_test_probs"
            with open(file + '.json', 'w') as json_file:
                json.dump(preds_dict, json_file)

