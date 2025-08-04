import torch
from framework.TransAppModel.TransApp import TransApp

def get_model_inst(m, win, dim_model, path_select_core=None, mode='pretraining', device='cuda'):

    TApp = TransApp(max_len=win, c_in=m,
                    mode=mode,
                    n_embed_blocks=1, 
                    encoding_type='noencoding',
                    n_encoder_layers=3,
                    kernel_size=5,
                    d_model=dim_model, pffn_ratio=2, n_head=4,
                    prenorm=True, norm="LayerNorm",
                    activation='gelu',
                    store_att=False, attn_dp_rate=0.2, head_dp_rate=0., dp_rate=0.2,
                    att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 
                               'learnable_scale_enc': False},
                    c_reconstruct=1, apply_gap=True, nb_class=2)
    
    if path_select_core is not None:
        if device == 'cpu':
            TApp.load_state_dict(torch.load(path_select_core, map_location=torch.device('cpu'))['model_state_dict'])
        else:
            TApp.load_state_dict(torch.load(path_select_core)['model_state_dict'])
    return TApp