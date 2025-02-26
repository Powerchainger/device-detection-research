#Importing libraries and files
from experiments.data_utils import *
from src.AD_Framework.Framework import *
from src.TransAppModel.TransApp import *
import json

def get_model_inst(m, win, dim_model, path_select_core=None):

    if path_select_core is not None:
        n_enc_layers = 5 if 'Large' in path_select_core else 3
    else:
        n_enc_layers = 3

    TApp = TransApp(max_len=win, c_in=m,
                    mode="classif",
                    n_embed_blocks=1, 
                    encoding_type='noencoding',
                    n_encoder_layers=n_enc_layers,
                    kernel_size=5,
                    d_model=dim_model, pffn_ratio=2, n_head=4,
                    prenorm=True, norm="LayerNorm",
                    activation='gelu',
                    store_att=False, attn_dp_rate=0.2, head_dp_rate=0., dp_rate=0.2,
                    att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                    c_reconstruct=1, apply_gap=True, nb_class=2)

    if path_select_core is not None:
        TApp.load_state_dict(torch.load(path_select_core)['model_state_dict'])

    return TApp

if __name__ == "__main__":
    #GLOBALS
    case_name = sys.argv[1]
    m = 5
    win = 1024 
    list_exo_variable = ['hours_cos', 'hours_sin', 'days_cos', 'days_sin']
    #--------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------
    # TODO: Create a dictionary to store the prediction probabilities of the models
    id_pred_prob = {} 
    #--------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------
    # TODO: Load the model for each case
    # TODO: First load the model for one case e.g. cooker_case
    #Get model path
    root = Path(os.getcwd()).resolve().parents[0]
    path = os.path.join(root, 'TransAppResults')
    #--------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------
    for seed_value in range(0,3):
        # Uncomment this to load a model with a specific seed. In this case 0
        model_path = os.path.join(path, str(case_name), 'TransAppPT96_1_' + str(seed_value) + '.pt')
        print(model_path)
        model = get_model_inst(m=m, win=win, dim_model=96, path_select_core=model_path)
        model.to('cuda')
        #--------------------------------------------------------------------------------------------------------------------------
        datas_tuple = CER_get_data_case(case_name, seed=seed_value, exo_variable=list_exo_variable,
                                                win=win, ratio_resample=0.8)
        # Entire curves data
        X_valid_voter = datas_tuple[8]
        y_valid_voter = datas_tuple[9]

        #Making torch datasets
        valid_voter = TSDataset(X_valid_voter, y_valid_voter)
        y = valid_voter[:][1].flatten()
        #Extracting the probabilities for each household
        for i, inst in enumerate(valid_voter):
            inst_ts, inst_label = inst
            inst_ts = np.reshape(inst_ts, (inst_ts.shape[0], m, inst_ts.shape[1]//m))
            n_obs_per_win = inst_ts.shape[-1] // win
            inst_ts = inst_ts[:, :, :n_obs_per_win*win]

            tmp = np.empty((n_obs_per_win, m, win))
            for im in range(m):
                tmp[:, im, :] = np.reshape(inst_ts[:, im, :], (n_obs_per_win, win))
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

                for ts, labels in loader:
                    model.eval()
                    # ===================variables=================== #
                    ts = Variable(ts.float()).to('cuda')
                    # ====================forward==================== #
                    logits = model(ts)
                    logits = nn.Softmax(dim=1)(logits)
                    if not logits_proba:
                        logits_proba = list(logits[:, 1].cpu().detach().numpy().ravel())
                    else:
                        logits_proba = logits_proba + list(logits[:, 1].cpu().detach().numpy().ravel())
                    #--------------------------------------------------------------------------------------------------------------------------
                    #--------------------------------------------------------------------------------------------------------------------------
                    # TODO : save vector of probabilities in float not numpy.float32
                logits_proba = [float(x) for x in logits_proba]
                rep_labels = [float(x) for x in rep_labels]
                id_pred_prob[i] = [logits_proba, rep_labels]
                    #--------------------------------------------------------------------------------------------------------------------------
                    #--------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------
        # TODO : save vector of probabilities
        key = f'Seed_{str(seed_value)}'
        file = f'./extracted_probabilities_seed/{str(case_name)}_validation_seed_{str(seed_value)}'
        with open(file + '.json', 'w') as json_file:
            json.dump(id_pred_prob, json_file)
        #--------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------         
