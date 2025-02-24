from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ckpt_path = r"models/best_phase2.pth"

for dataset_name in ["MSRS"]:
    print("\n" * 2 + "=" * 80)
    model_name = "LIGFusion    "
    print("The test result of " + dataset_name + ' :')
    test_folder = os.path.join('test_img', dataset_name)
    test_out_folder = os.path.join('test_result', dataset_name)
    os.makedirs(test_out_folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    with torch.no_grad():
        img_list = os.listdir(os.path.join(test_folder, "ir"))
        total_imgs = len(img_list)
        progress_bar = tqdm(total=total_imgs, desc=f'Processing {dataset_name}',
                            unit='img', dynamic_ncols=True,
                            bar_format='{l_bar}{bar} | {n_fmt}/{total_fmt} [ {remaining} ]')

        for img_name in img_list:

            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_IR = torch.FloatTensor(data_IR).cuda()

            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            data_VIS_YCrCb = cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb)
            data_VIS_Y, data_VIS_Cr, data_VIS_Cb = cv2.split(data_VIS_YCrCb)
            data_VIS_Y_orig = data_VIS_Y.copy()
            data_VIS_Y = data_VIS_Y[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_Y = torch.FloatTensor(data_VIS_Y).cuda()

            feature_I_B, feature_I_D, feature_I = Encoder(data_IR, is_vis=False, training_phase=2)
            feature_V_B, feature_V_D, feature_V, W_ir, W_vis = Encoder(data_VIS_Y, is_vis=True, training_phase=2)
            feature_F_B = BaseFuseLayer(W_vis * feature_V_B + W_ir * feature_I_B)
            feature_F_D = DetailFuseLayer(W_vis * feature_V_D + W_ir * feature_I_D)

            data_Fuse, _ = Decoder(data_VIS_Y, feature_F_B, feature_F_D)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))

            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = np.clip(fi, 0, 255).astype(np.uint8)
            vis_Y_float = data_VIS_Y_orig.astype(np.float32)

            sigma1, sigma2 = 0.5, 1.0
            g1 = cv2.GaussianBlur(vis_Y_float, (0, 0), sigmaX=sigma1)
            g2 = cv2.GaussianBlur(vis_Y_float, (0, 0), sigmaX=sigma2)
            dog = g1 - g2
            alpha = 0.1
            fi = fi.astype(np.float32) + alpha * dog
            fi = np.clip(fi, 0, 255).astype(np.uint8)

            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
            progress_bar.update(1)

        progress_bar.close()