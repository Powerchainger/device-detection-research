from framework.AD_Framework.Framework import TSDataset, self_pretrainer
from framework.utils.losses import MaskedMSELoss
import torch

def launch_pretraining(model, 
                       save_path,
                       X_train,
                       GeomMask,
                       dict_params):
    
    pretraining_dataset = TSDataset(X_train, scaler=True, scale_dim=[0])
    train_loader = torch.utils.data.DataLoader(pretraining_dataset, batch_size=dict_params['batch_size'], shuffle=True)
    model_pretrainer = self_pretrainer(model,                                     
                                       train_loader, valid_loader=None,
                                       learning_rate=dict_params['lr'], weight_decay=dict_params['wd'],
                                       name_scheduler='CosineAnnealingLR',
                                       dict_params_scheduler={'T_max': dict_params['epochs'], 'eta_min': 1e-6},
                                       warmup_duration=None,
                                       criterion=MaskedMSELoss(type_loss='L1'), mask=GeomMask,
                                       device="cuda", all_gpu=True,
                                       verbose=True, plotloss=False, 
                                       save_fig=False, path_fig=None,
                                       save_only_core=False,
                                       save_checkpoint=True, path_checkpoint=save_path)

    model_pretrainer.train(dict_params['epochs'])
