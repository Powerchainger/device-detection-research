from config.config import CHECKPOINT_DIR, CASES, DATA_PARAMS, TRAINING_PARAMS
from utils.data_utils import create_dir, CER_get_data_case
from framework.utils.TransApp_utils import get_model_inst
from framework.utils.training import launch_training
import json, os

path_core = str(CHECKPOINT_DIR) + "\\TransApp" + str(TRAINING_PARAMS['dim_model']) + ".pt"
dir_path = create_dir(str(CHECKPOINT_DIR) + "\\case_headers")
voter_path  = create_dir(str(CHECKPOINT_DIR) + "\\voter_metrics")

def run_training(data_name='x_residential_25728.csv', extra_name='extra_25728'):
    train_dict = {"lr": TRAINING_PARAMS['lr'], 
                    "wd": 1e-3, #Different weight decay for training 
                    "batch_size": TRAINING_PARAMS['batch_size'], 
                    "epochs": TRAINING_PARAMS['epochs'],
                    "p_es": TRAINING_PARAMS['p_es'],
                    "p_rlr": TRAINING_PARAMS['p_rlr'],
                    "n_warmup_epochs": TRAINING_PARAMS['n_warmup_epochs']}    
    for case in CASES:        
        save_path = str(dir_path) + "\\" + case
        data_tuple = CER_get_data_case(case_name=case, exo_variable=DATA_PARAMS, 
                                       win=TRAINING_PARAMS['window_size'],
                                       data_name=data_name,
                                       extra_name=extra_name)
        model = get_model_inst(TRAINING_PARAMS['input_dim'],
                        TRAINING_PARAMS['window_size'],
                        TRAINING_PARAMS['dim_model'],
                        path_select_core=path_core,
                        mode='classification')
        voter_dict = launch_training(model,
                        save_path,
                        data_tuple,
                        train_dict,
                        m=TRAINING_PARAMS['input_dim'],
                        win=TRAINING_PARAMS['window_size'],
                        seed=0 
                        )
        # Save best quantile value and f1_macro
        with open(os.path.join(voter_path, f"{case}.json"), "w") as f:
            json.dump(voter_dict, f)