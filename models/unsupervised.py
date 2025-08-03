from config.config import CHECKPOINT_DIR, DATA_DIR, DATA_PARAMS, TRAINING_PARAMS
from utils.data_utils import create_dir, CER_get_data_pretraining
from framework.utils.TransApp_utils import get_model_inst
from framework.utils.pretraining import launch_pretraining

def run_pretraining(input_path=None, data_name='x_residential_25728.csv'):
    dir = create_dir(str(CHECKPOINT_DIR))
    path = dir + "\\TransApp" + str(TRAINING_PARAMS['dim_model'])
    X_train = CER_get_data_pretraining(win=TRAINING_PARAMS['wd'], 
                                        exo_variable=DATA_PARAMS,
                                        input_path=input_path,
                                        data_name=data_name)
        
    model = get_model_inst(TRAINING_PARAMS['input_dim'], 
                        TRAINING_PARAMS['window_size'], 
                        TRAINING_PARAMS['dim_model'],
                        mode='pretraining')
    
    train_params = {"lr": TRAINING_PARAMS['lr'], 
                    "wd": TRAINING_PARAMS['wd'], 
                    "batch_size": TRAINING_PARAMS['batch_size'], 
                    "epochs": TRAINING_PARAMS['epochs']}
    
    launch_pretraining(model,
                    path,
                    X_train, 
                    TRAINING_PARAMS['GeoMask'],
                    train_params)
