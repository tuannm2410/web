import torch
import argparse
import pickle
from argparse import Namespace

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description
import os

path = os.path.dirname(__file__)

def create_description(img_path):
    model_dim = 512
    N_enc = 3
    N_dec = 3
    max_seq_len = 74
    beam_size = 5
    load_path = path + '/checkpoint.pth'  
    image_paths = img_path

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim= model_dim,
                           N_enc= N_enc,
                           N_dec= N_dec,
                           dropout=0.0,
                           drop_args=drop_args)

    with open(path + '/data1/demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    img_size = 384
    '''model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=max_seq_len, drop_args=model_args.drop_args,
                                rank='cpu')'''
    
    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                    swin_embed_dim=128, swin_depths=[2, 2, 18, 2], swin_num_heads=[4, 8, 16, 32],
                                    swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                    swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                    swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                    swin_use_checkpoint=False,
                                    final_swin_dim=1024,

                                    d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                    N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                    num_exp_enc_list=[32, 64, 128, 256, 512],
                                    num_exp_dec=16,
                                    output_word2idx=coco_tokens['word2idx_dict'],
                                    output_idx2word=coco_tokens['idx2word_list'],
                                    max_seq_len=max_seq_len, drop_args=model_args.drop_args,
                                    rank='cpu')


    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    input_images = preprocess_image(image_paths, img_size)
    image = input_images
    beam_search_kwargs = {'beam_size': beam_size,
                            'beam_max_seq_len': max_seq_len,
                            'sample_or_max': 'max',
                            'how_many_outputs': 1,
                            'sos_idx': sos_idx,
                            'eos_idx': eos_idx}
    with torch.no_grad():
        pred, _ = model(enc_x=image,
                        enc_x_num_pads=[0],
                        mode='beam_search', **beam_search_kwargs)
    pred = tokens2description(pred[0][0], coco_tokens['idx2word_list'], sos_idx, eos_idx)

    return pred

'''text = create_description('C:/Users/ADMIN/Desktop/web/Image2Text/cat_girl.jpg')
print(text)'''
