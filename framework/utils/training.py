from framework.AD_Framework.Framework import TSDataset, AD_Framework, getmetrics

import torch
import torch.nn as nn


def launch_training(model,
                    save_path,
                    datas_tuple,
                    dict_params,
                    m=5,
                    win=1024,
                    seed=0):
    """
    Launch model training

    Input :
    - model : model instance
    - save_path : path to save model / case
    - m : number of variable of the MTS
    - win : window size of subsequences
    - datas_tuple : [X_train, y_train, ... X_test_voter , y_test_voter]
    - dict_params : dictionary of parameters
    """

    # Scliced data
    X_train = datas_tuple[0]
    y_train = datas_tuple[1]
    X_valid = datas_tuple[2]
    y_valid = datas_tuple[3]
    X_test  = datas_tuple[4]
    y_test  = datas_tuple[5]

    # Entire curves data
    X_train_voter = datas_tuple[6]
    y_train_voter = datas_tuple[7]
    X_valid_voter = datas_tuple[8]
    y_valid_voter = datas_tuple[9]
    X_test_voter  = datas_tuple[10]
    y_test_voter  = datas_tuple[11]

    # Dataset
    train_dataset = TSDataset(X_train, y_train, scaler=True, scale_dim=[0])
    valid_dataset = TSDataset(X_valid, y_valid, scaler=True, scale_dim=[0])
    test_dataset  = TSDataset(X_test, y_test,   scaler=True, scale_dim=[0])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dict_params['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)
    model_trainer = AD_Framework(model, seed,
                                 train_loader=train_loader, valid_loader= valid_loader,
                                 learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                 criterion=nn.CrossEntropyLoss(),
                                 patience_es=dict_params['p_es'], patience_rlr=dict_params['p_rlr'],
                                 f_metrics=getmetrics(),
                                 n_warmup_epochs=dict_params['n_warmup_epochs'],
                                 scale_by_subseq_in_voter=True, scale_dim=[0],
                                 verbose=True, plotloss=False, 
                                 save_fig=False, path_fig=None,
                                 device="cuda", all_gpu=True, # all_gpu - True only when multiple GPUs are present
                                 save_checkpoint=True, path_checkpoint=save_path) # save_checkpoint = True?? How many are saved?
    model_trainer.train(dict_params['epochs'])

    #============ eval last model ============#
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1), mask='test_metrics_lastmodel')

    #============ restore best weight and evaluate ============#    
    model_trainer.restore_best_weights()
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))

    # ======================== THIS IF WE WANT TO ADD THE VOTER AS WELL ======================== #
    
    
    # ============= find best quantile on valid dataset =============#
    voter_train_metrics = model_trainer.ADFFindBestQuantile(TSDataset(X_valid_voter, y_valid_voter), m=m, win=win)
    voter_test_metrics = model_trainer.ADFvoter_proba(TSDataset(X_test_voter, y_test_voter), m=m, win=win)
    voter_f1_macro = voter_test_metrics["F1_SCORE_MACRO"]
    voter_quantile = voter_train_metrics["quantile"]    
    # ========================================================================================== #
    return {"F1_SCORE_MACRO": voter_f1_macro, "quantile": voter_quantile}
