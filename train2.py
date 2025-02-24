import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc
import os
import sys
import time
import datetime
import kornia
from collections import OrderedDict

def remove_module_prefix(state_dict):

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict

def phase_ii_train():

    criteria_fusion = Fusionloss()
    num_epochs = 40 
    lr = 1e-4  
    weight_decay = 1e-4  
    batch_size = 32
    coeff_decomp = 2.0  
    coeff_mse_loss_Fuse = 2.0  
    coeff_ssim_loss_Fuse = 2.0  
    coeff_tv = 5.0  
    clip_grad_norm_value = 5.0  
    optim_step = 20
    optim_gamma = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DIDF_Encoder = Restormer_Encoder().to(device)
    DIDF_Decoder = Restormer_Decoder().to(device)
    BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)
    checkpoint = torch.load("models/best_phase1.pth", map_location=device)
    print(checkpoint.keys())
    checkpoint['DIDF_Encoder'] = remove_module_prefix(checkpoint['DIDF_Encoder'])
    checkpoint['DIDF_Decoder'] = remove_module_prefix(checkpoint['DIDF_Decoder'])
    checkpoint['BaseFeatureExtraction'] = remove_module_prefix(checkpoint['BaseFeatureExtraction'])
    checkpoint['DetailFeatureExtraction'] = remove_module_prefix(checkpoint['DetailFeatureExtraction'])
    DIDF_Encoder.load_state_dict(checkpoint['DIDF_Encoder'], strict=False)
    DIDF_Decoder.load_state_dict(checkpoint['DIDF_Decoder'], strict=False)
    BaseFuseLayer.load_state_dict(checkpoint['BaseFeatureExtraction'], strict=False)
    DetailFuseLayer.load_state_dict(checkpoint['DetailFeatureExtraction'], strict=False)
    print("Loaded Phase I weights successfully")
    DIDF_Encoder = nn.DataParallel(DIDF_Encoder)
    DIDF_Decoder = nn.DataParallel(DIDF_Decoder)
    BaseFuseLayer = nn.DataParallel(BaseFuseLayer)
    DetailFuseLayer = nn.DataParallel(DetailFuseLayer)
    optimizer1 = optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer3 = optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer4 = optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
    scheduler4 = optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
    MSELoss = nn.MSELoss(reduction='mean')
    L1Loss = nn.L1Loss(reduction='mean')
    Loss_ssim = kornia.losses.SSIM(11, reduction='mean')
    trainloader = DataLoader(
        H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    best_loss = float('inf')
    prev_time = time.time()
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (data_VIS, data_IR, data_VIS_original) in enumerate(trainloader):
            data_VIS = data_VIS.to(device, non_blocking=True)
            data_IR = data_IR.to(device, non_blocking=True)
            data_VIS_original = data_VIS_original.to(device, non_blocking=True)
            DIDF_Encoder.train()
            DIDF_Decoder.train()
            BaseFuseLayer.train()
            DetailFuseLayer.train()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            outputs_vis = DIDF_Encoder(data_VIS_original, is_vis=True, training_phase=2)
            feature_V_B, feature_V_D, feature_V, W_ir, W_vis = outputs_vis
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR, is_vis=False, training_phase=2)[:3]

            fused_feature_B = W_ir * feature_I_B + W_vis * feature_V_B
            fused_feature_D = W_ir * feature_I_D + W_vis * feature_V_D

            feature_F_B = BaseFuseLayer(fused_feature_B)
            feature_F_D = DetailFuseLayer(fused_feature_D)

            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)
            mse_loss_Fuse = coeff_ssim_loss_Fuse * Loss_ssim(data_Fuse, data_VIS_original) + coeff_mse_loss_Fuse * MSELoss(data_Fuse, data_VIS_original)
            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

            weighted_fusionloss = W_ir * fusionloss + W_vis * fusionloss
            weighted_mse_loss_Fuse = W_ir * mse_loss_Fuse + W_vis * mse_loss_Fuse

            loss = weighted_fusionloss.mean() + coeff_decomp * loss_decomp.mean() + weighted_mse_loss_Fuse.mean()
            loss.backward()

            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            epoch_loss += loss.item()
            batches_done = epoch * len(trainloader) + i
            batches_left = num_epochs * len(trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %.6f] ETA: %s"
                % (
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    len(trainloader),
                    loss.item(),
                    str(time_left).split('.')[0],
                )
            )
            sys.stdout.flush()


        epoch_loss /= len(trainloader)
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")


        if epoch_loss < best_loss:
            best_loss = epoch_loss

            torch.save({
                'DIDF_Encoder': DIDF_Encoder.state_dict(),
                'DIDF_Decoder': DIDF_Decoder.state_dict(),
                'BaseFuseLayer': BaseFuseLayer.state_dict(),
                'DetailFuseLayer': DetailFuseLayer.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'optimizer3': optimizer3.state_dict(),
                'optimizer4': optimizer4.state_dict(),
                'scheduler1': scheduler1.state_dict(),
                'scheduler2': scheduler2.state_dict(),
                'scheduler3': scheduler3.state_dict(),
                'scheduler4': scheduler4.state_dict(),
                'epoch': epoch + 1,
                'loss': epoch_loss,
            }, "models/best_newtrainloss120.pth")
            print(f"Best model saved at epoch {epoch + 1} with loss: {epoch_loss:.6f}")


        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()

if __name__ == '__main__':

    os.makedirs("models", exist_ok=True)
    phase_ii_train()